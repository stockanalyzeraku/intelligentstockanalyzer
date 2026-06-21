"""
Orchestration layer: scrape a company from screener.in and persist it into
SQLite, using the typed tables defined in db.py.

This is the main entry point most calling code should use:

    from codebase.financials.ingest import ingest_company
    result = ingest_company("RELIANCE")

It is idempotent / safe to re-run: companies are upserted by
screener_symbol, and statement rows are upserted by
(company_id, line_item, period_label) so re-scraping a company updates
existing rows rather than duplicating them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from datetime import datetime, timezone

from codebase.financials import db
from codebase.financials.db import STATEMENT_TABLES
from codebase.financials.scraper import ScrapeError, scrape_company
from config import CONFIG
    

# Map a "Mon YYYY" period label (e.g. "Mar 2024") to month number, used to
# derive a real ISO fiscal_year_end date for the typed column.
_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _period_to_iso_date(period_label):
    """'Mar 2024' -> '2024-03-31'  (best-effort; returns None if unparsable)."""
    m = re.match(r"^([A-Za-z]{3})\s+(\d{4})$", period_label.strip())
    if not m:
        return None
    mon, year = m.group(1), int(m.group(2))
    month_num = _MONTH_MAP.get(mon)
    if not month_num:
        return None
    # Last day of month - simple lookup, good enough for fiscal year-end dates.
    last_day = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[month_num]
    if month_num == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        last_day = 29
    return f"{year:04d}-{month_num:02d}-{last_day:02d}"


def _upsert_company(conn, company_info, scraped_at):
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM companies WHERE screener_symbol = ?",
        (company_info.screener_symbol,),
    )
    existing = cur.fetchone()

    if existing:
        company_id = existing["id"]
        cur.execute(
            """
            UPDATE companies
               SET input_symbol = ?, company_name = ?, nse_symbol = ?,
                   bse_code = ?, source_url = ?, last_scraped_at = ?
             WHERE id = ?
            """,
            (
                company_info.input_symbol,
                company_info.company_name,
                company_info.nse_symbol,
                company_info.bse_code,
                company_info.source_url,
                scraped_at,
                company_id,
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO companies
                (input_symbol, screener_symbol, company_name, nse_symbol,
                 bse_code, source_url, last_scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_info.input_symbol,
                company_info.screener_symbol,
                company_info.company_name,
                company_info.nse_symbol,
                company_info.bse_code,
                company_info.source_url,
                scraped_at,
            ),
        )
        company_id = cur.lastrowid

    return company_id


def _table_name_for_statement_key(statement_key):
    for table_name, meta in STATEMENT_TABLES.items():
        if meta["statement_key"] == statement_key:
            return table_name
    raise ValueError(f"No typed table registered for statement_key='{statement_key}'")


def _upsert_statement_rows(conn, company_id, statement_table, scraped_at):
    table_name = _table_name_for_statement_key(statement_table.statement_key)
    cur = conn.cursor()

    rows_written = 0
    for row in statement_table.rows:
        for period_label, value in row.values.items():
            fiscal_year_end = _period_to_iso_date(period_label)
            cur.execute(
                f"""
                INSERT INTO {table_name}
                    (company_id, line_item, period_label, fiscal_year_end, value, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_id, line_item, period_label)
                DO UPDATE SET value = excluded.value,
                              fiscal_year_end = excluded.fiscal_year_end,
                              scraped_at = excluded.scraped_at
                """,
                (company_id, row.line_item, period_label, fiscal_year_end, value, scraped_at),
            )
            rows_written += 1
    return rows_written


def ingest_company(symbol, years_to_keep=None, db_path=None):
    """Scrape `symbol` from screener.in and persist it to SQLite.

    Returns a summary dict:
        {
            "company_id": int,
            "company_name": str,
            "screener_symbol": str,
            "statements_written": {"profit_loss": 5, "balance_sheet": 10, "cash_flow": 6},
            "periods": ["Mar 2022", ..., "Mar 2026"],
        }

    Raises ScrapeError if the symbol can't be scraped (e.g. not found).
    """
    db.init_schema(db_path)

    scraped = scrape_company(symbol, years_to_keep=years_to_keep)
    scraped_at = datetime.now(timezone.utc).isoformat()

    statements_written = {}
    periods = []

    with db.get_connection(db_path) as conn:
        company_id = _upsert_company(conn, scraped.company, scraped_at)

        for statement in scraped.statements:
            n = _upsert_statement_rows(conn, company_id, statement, scraped_at)
            statements_written[statement.statement_key] = n
            if len(statement.periods) > len(periods):
                periods = statement.periods

        db.refresh_line_item_catalogue(conn)

    return {
        "company_id": company_id,
        "company_name": scraped.company.company_name,
        "screener_symbol": scraped.company.screener_symbol,
        "nse_symbol": scraped.company.nse_symbol,
        "bse_code": scraped.company.bse_code,
        "statements_written": statements_written,
        "periods": periods,
    }


def ingest_many(symbols, years_to_keep=None, db_path=None, stop_on_error=False):
    """Ingest multiple symbols in one go. Returns list of per-symbol results,
    each either the success dict from ingest_company() or
    {"symbol": ..., "error": str} on failure.
    """
    results = []
    for symbol in symbols:
        try:
            result = ingest_company(symbol, years_to_keep=years_to_keep, db_path=db_path)
            results.append(result)
        except ScrapeError as exc:
            results.append({"symbol": symbol, "error": str(exc)})
            if stop_on_error:
                break
    return results

if __name__ == "__main__":
    result = ingest_company("KALYANKJIL",5,CONFIG.DB_PATH)
    print(result)


