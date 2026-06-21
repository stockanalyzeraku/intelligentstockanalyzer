"""
Agent-facing discovery & dynamic read API.

This module is the intended entry point for any agent (or downstream code)
that wants to explore "what tables exist and what do they mean" and then
read data WITHOUT hardcoding any knowledge of the schema up front.

Typical agent flow:
    1. list_tables()                       -> what logical tables exist?
    2. describe_table("statement_cash_flow") -> what columns does it have?
    3. list_companies() / find_company("RELIANCE") -> resolve a company_id
    4. list_line_items("statement_cash_flow", company_id=...) -> what rows exist?
    5. get_statement("statement_cash_flow", company_id=...)   -> get the data
       (as tidy long rows, or pivoted wide by year)

Every function here returns plain dicts/lists (JSON-serialisable), so it is
safe to expose directly as tool calls to an LLM agent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from codebase.financials import db


def list_tables(db_path=None):
    """Return every logical table in the database with its category and
    description, straight from the _meta_tables discovery table.
    """
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT table_name, category, description FROM _meta_tables ORDER BY category, table_name"
        ).fetchall()
        return [dict(r) for r in rows]


def describe_table(table_name, db_path=None):
    """Return the column-level schema for one logical table, from
    _meta_columns, plus the row count currently in that table.
    """
    with db.get_connection(db_path) as conn:
        cols = conn.execute(
            "SELECT column_name, data_type, description FROM _meta_columns WHERE table_name = ?",
            (table_name,),
        ).fetchall()
        if not cols:
            raise ValueError(f"Unknown table '{table_name}'. Call list_tables() to see valid names.")

        try:
            count_row = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()
            row_count = count_row["n"]
        except Exception:
            row_count = None

        return {
            "table_name": table_name,
            "row_count": row_count,
            "columns": [dict(c) for c in cols],
        }


def list_companies(db_path=None):
    """Return every company currently stored, for an agent to pick a company_id from."""
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id AS company_id, company_name, screener_symbol, nse_symbol,
                   bse_code, last_scraped_at
              FROM companies
             ORDER BY company_name
            """
        ).fetchall()
        return [dict(r) for r in rows]


def find_company(symbol, db_path=None):
    """Resolve a ticker/symbol (NSE symbol, BSE code, or screener symbol) to
    the stored company record. Returns None if not found - caller should
    then call ingest_company(symbol) to scrape it first.
    """
    symbol_norm = symbol.strip().upper()
    with db.get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id AS company_id, company_name, screener_symbol, nse_symbol,
                   bse_code, source_url, last_scraped_at
              FROM companies
             WHERE UPPER(screener_symbol) = ?
                OR UPPER(nse_symbol) = ?
                OR UPPER(bse_code) = ?
                OR UPPER(input_symbol) = ?
            """,
            (symbol_norm, symbol_norm, symbol_norm, symbol_norm),
        ).fetchone()
        return dict(row) if row else None


def list_statement_tables(db_path=None):
    """Return just the names of the financial-statement typed tables
    (excludes companies + the _meta_* discovery tables).
    """
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT table_name FROM _meta_tables WHERE category = 'financial_statement' ORDER BY table_name"
        ).fetchall()
        return [r["table_name"] for r in rows]


def list_line_items(table_name, company_id=None, db_path=None):
    """Discover which line_item labels exist for a given statement table
    (optionally scoped to one company, since not every company discloses
    every line item, e.g. banks vs manufacturers).
    """
    valid_tables = list_statement_tables(db_path)
    if table_name not in valid_tables:
        raise ValueError(f"Unknown statement table '{table_name}'. Valid options: {valid_tables}")

    with db.get_connection(db_path) as conn:
        if company_id is not None:
            rows = conn.execute(
                f"SELECT DISTINCT line_item FROM {table_name} WHERE company_id = ? ORDER BY line_item",
                (company_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT line_item FROM _meta_line_items WHERE table_name = ? ORDER BY line_item",
                (table_name,),
            ).fetchall()
        return [r["line_item"] for r in rows]


def get_statement(table_name, company_id, line_items=None, periods=None,
                   pivot=False, db_path=None):
    """Read statement data for one company out of a typed table.

    Parameters
    ----------
    table_name : str
        One of the names returned by list_statement_tables(), e.g.
        "statement_profit_loss".
    company_id : int
        From find_company() / list_companies().
    line_items : list[str], optional
        Restrict to specific row labels, e.g. ["Sales", "Net Profit"].
        If omitted, all available line items are returned.
    periods : list[str], optional
        Restrict to specific period labels, e.g. ["Mar 2024", "Mar 2025"].
        If omitted, all available years are returned.
    pivot : bool
        If False (default): returns tidy long rows -
            [{"line_item": "Sales", "period_label": "Mar 2024", "value": 899041.0}, ...]
        If True: returns one dict per line_item with years as keys -
            [{"line_item": "Sales", "Mar 2024": 899041.0, "Mar 2025": 962820.0}, ...]

    Returns
    -------
    list[dict]
    """
    valid_tables = list_statement_tables(db_path)
    if table_name not in valid_tables:
        raise ValueError(f"Unknown statement table '{table_name}'. Valid options: {valid_tables}")

    query = f"""
        SELECT line_item, period_label, fiscal_year_end, value
          FROM {table_name}
         WHERE company_id = ?
    """
    params = [company_id]

    if line_items:
        placeholders = ",".join("?" for _ in line_items)
        query += f" AND line_item IN ({placeholders})"
        params.extend(line_items)

    if periods:
        placeholders = ",".join("?" for _ in periods)
        query += f" AND period_label IN ({placeholders})"
        params.extend(periods)

    query += " ORDER BY line_item, fiscal_year_end"

    with db.get_connection(db_path) as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    if not pivot:
        return rows

    pivoted = {}
    order = []
    for r in rows:
        li = r["line_item"]
        if li not in pivoted:
            pivoted[li] = {"line_item": li}
            order.append(li)
        pivoted[li][r["period_label"]] = r["value"]

    return [pivoted[li] for li in order]


def get_all_statements_for_company(company_id, pivot=True, db_path=None):
    """Convenience wrapper: fetch P&L + Balance Sheet + Cash Flow for one
    company in a single call. Returns a dict keyed by table_name.
    """
    result = {}
    for table_name in list_statement_tables(db_path):
        result[table_name] = get_statement(table_name, company_id, pivot=pivot, db_path=db_path)
    return result


def run_raw_query(sql, params=(), db_path=None):
    """Escape hatch for an agent that wants to run arbitrary read-only SQL
    after discovering the schema via the functions above.

    SAFETY: only SELECT statements are permitted; anything else raises
    ValueError. This is a discovery/debug tool, not a general SQL executor.
    """
    stripped = sql.strip().lower()
    if not stripped.startswith("select"):
        raise ValueError("run_raw_query only permits SELECT statements.")

    with db.get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
