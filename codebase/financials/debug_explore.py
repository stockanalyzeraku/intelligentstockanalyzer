"""
Standalone debug/explore module for the financials package.

This file is intentionally separate from the library code (db.py,
discovery.py, ingest.py) so you can poke at the database and schema
without digging through the other modules - useful while developing,
debugging scraper changes, or just inspecting what's in financials.db.

NOT a CLI. Import the functions you need and call them directly with
normal Python arguments. Every function returns a plain dict/list (no
printing, no argparse) so you can use the result in code, print it
yourself, or just inspect it in a REPL/notebook.

USAGE
-----
    from codebase.financials.debug_explore import (
        tables, describe, companies, ingest, dump, line_items, sql
    )

    tables()                                  # -> list[dict] of every table + description
    describe("statement_profit_loss")         # -> dict of columns for one table
    companies()                               # -> list[dict] of every ingested company
    ingest("KALYANKJIL")                      # -> scrape + store, dict summary
    dump("KALYANKJIL")                        # -> dict of all 3 statements, pivoted by year
    dump("KALYANKJIL", table="statement_cash_flow")  # -> just one statement
    dump("KALYANKJIL", long=True)             # -> tidy long rows instead of pivoted
    line_items("statement_balance_sheet")                    # -> list[str], any company
    line_items("statement_balance_sheet", symbol="KALYANKJIL")  # -> list[str], scoped to one company
    sql("SELECT * FROM companies")            # -> list[dict], SELECT-only

If you just run this file directly (``python3 debug_explore.py``), the
__main__ block at the bottom calls a few of these and prints the result -
edit that block to try out whatever you're debugging.

Every function raises a plain Exception (ScrapeError or ValueError) on
failure rather than printing an error and exiting - since this is meant to
be called from your own code now, not a terminal, callers should catch
these themselves if they want to handle failures gracefully.
"""

import sys
from pathlib import Path

# Allow importing/running this file directly regardless of cwd, by making
# sure the parent of codebase/ is on sys.path.

from codebase.financials import db, discovery  # noqa: E402
from codebase.financials.ingest import ingest_company  # noqa: E402
from codebase.financials.scraper import ScrapeError  # noqa: E402

__all__ = ["tables", "describe", "companies", "ingest", "dump", "line_items", "sql"]


def tables():
    """List every logical table in the database with its category and description.

    Returns
    -------
    list[dict]
    """
    db.init_schema()
    return discovery.list_tables()


def describe(table_name):
    """Show columns (name, type, description) + row count for one table.

    Parameters
    ----------
    table_name : str
        e.g. "statement_profit_loss", "companies", "_meta_units".
        Run tables() first if you're not sure of the exact name.

    Returns
    -------
    dict

    Raises
    ------
    ValueError if table_name doesn't exist.
    """
    db.init_schema()
    return discovery.describe_table(table_name)


def companies():
    """List every company currently stored in the DB.

    Returns
    -------
    list[dict]
    """
    db.init_schema()
    return discovery.list_companies()


def ingest(symbol, years=None):
    """Scrape `symbol` from screener.in and store it (or re-scrape/update it
    if already stored).

    Parameters
    ----------
    symbol : str
        NSE trading symbol as used by screener.in, e.g. "RELIANCE", "KALYANKJIL".
    years : int, optional
        How many recent annual years to keep. Defaults to 5.

    Returns
    -------
    dict
        Summary: company_id, company_name, screener_symbol, nse_symbol,
        bse_code, statements_written, periods.

    Raises
    ------
    ScrapeError if the symbol can't be found/scraped on screener.in.
    """
    db.init_schema()
    return ingest_company(symbol, years_to_keep=years)


def _resolve_company(symbol):
    company = discovery.find_company(symbol)
    if not company:
        raise ValueError(
            f"No stored data for '{symbol}' yet. Call ingest('{symbol}') first."
        )
    return company


def dump(symbol, table=None, long=False):
    """Get stored statement data for a company.

    Parameters
    ----------
    symbol : str
        Company symbol previously passed to ingest(), e.g. "KALYANKJIL".
    table : str, optional
        Limit to one statement table, e.g. "statement_cash_flow".
        If omitted, returns all three statements.
    long : bool
        If False (default): values pivoted by year, one dict per line_item
            with a "unit" key and one key per period, e.g.
            {"line_item": "Sales", "unit": "INR_CRORE", "Mar 2025": 962820.0, ...}
        If True: tidy long rows instead, one dict per (line_item, period), e.g.
            {"line_item": "Sales", "period_label": "Mar 2025", "value": 962820.0,
             "fiscal_year_end": "2025-03-31", "unit": "INR_CRORE"}

    Returns
    -------
    dict
        {"company": {...}, "statement_profit_loss": [...], ...}
        (or just {"company": {...}, table: [...]} if `table` was given)

    Raises
    ------
    ValueError if the company hasn't been ingested yet, or `table` is unknown.
    """
    db.init_schema()
    company = _resolve_company(symbol)
    company_id = company["company_id"]

    if table:
        data = discovery.get_statement(table, company_id, pivot=not long)
        return {"company": company, table: data}

    data = discovery.get_all_statements_for_company(company_id, pivot=not long)
    return {"company": company, **data}


def line_items(table_name, symbol=None):
    """List which line_item labels (row names) exist for a statement table.

    Parameters
    ----------
    table_name : str
        e.g. "statement_balance_sheet".
    symbol : str, optional
        If given, scopes to line items actually present for that company
        (not every company discloses every line item, e.g. banks vs
        manufacturers). If omitted, lists every line item ever seen across
        all ingested companies.

    Returns
    -------
    dict
        {"table": table_name, "company": company_or_None, "line_items": [...]}

    Raises
    ------
    ValueError if table_name is unknown, or symbol hasn't been ingested.
    """
    db.init_schema()
    company = None
    company_id = None
    if symbol:
        company = _resolve_company(symbol)
        company_id = company["company_id"]
    items = discovery.list_line_items(table_name, company_id=company_id)
    return {"table": table_name, "company": company, "line_items": items}


def sql(query, params=()):
    """Run an arbitrary read-only SQL query.

    Parameters
    ----------
    query : str
        Must start with SELECT - anything else raises ValueError.
    params : tuple, optional
        Positional parameters for "?" placeholders in `query`.

    Returns
    -------
    list[dict]
    """
    db.init_schema()
    return discovery.run_raw_query(query, params)


if __name__ == "__main__":
    # Quick manual smoke test when running this file directly. Edit this
    # block freely while debugging - it has no effect on the importable
    # functions above.
    import json

    print(json.dumps(dump("KALYANKJIL"), indent=2, default=str))
