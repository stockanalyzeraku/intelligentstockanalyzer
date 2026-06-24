"""
SQLite storage layer for scraped financial data.

STORAGE STYLE: hybrid (typed tables + metadata/discovery tables)
------------------------------------------------------------------
1. Typed tables hold the actual data with real columns and types:
     - companies                 : one row per company (identity/master data)
     - statement_profit_loss     : P&L line items, long/tidy format
     - statement_balance_sheet   : Balance Sheet line items, long/tidy format
     - statement_cash_flow       : Cash Flow line items, long/tidy format

   Each statement table uses a long ("tidy") layout:
       company_id | line_item | period_label | value | ...
   rather than one column per year. This is deliberate:
     - Screener's available years shift every year (a fresh scrape next
       year adds "Mar 2027" etc.) - a wide table would need a schema
       migration (ALTER TABLE ADD COLUMN) every single year. The long
       format never needs migration: new years are just new rows.
     - It is trivially and uniformly queryable/filterable by an agent or by
       SQL ("get me Sales for company X between 2022 and 2025") without
       needing to know in advance which year-columns exist.

2. Metadata/discovery tables let a caller (human or agent) discover the
   schema and its meaning purely by querying SQLite - no need to read this
   source file:
     - _meta_tables     : one row per logical table, with a human description
     - _meta_columns    : one row per column of each logical table, with
                          type and description
     - _meta_line_items : catalogue of every distinct line_item label seen
                          per statement table, so an agent can discover
                          *what data exists* before writing a query.

This module only deals with schema + low-level read/write. Higher-level
orchestration (scrape -> store) lives in ingest.py. The standalone
explorer/debug script is debug_explore.py.
"""
import sys
import os

import sqlite3
from contextlib import contextmanager

from codebase.financials import config

# ---------------------------------------------------------------------------
# Logical schema description - single source of truth for both the actual
# CREATE TABLE statements AND the metadata rows we seed into _meta_tables /
# _meta_columns. Keeping this in one place means the discovery tables can
# never drift out of sync with the real schema.
# ---------------------------------------------------------------------------

STATEMENT_TABLES = {
    "statement_profit_loss": {
        "statement_key": "profit_loss",
        "description": (
            "Annual consolidated Profit & Loss statement line items "
            "(Sales, Expenses, Operating Profit, Net Profit, EPS, etc.) "
            "scraped from screener.in. One row per (company, line item, year)."
        ),
    },
    "statement_balance_sheet": {
        "statement_key": "balance_sheet",
        "description": (
            "Annual consolidated Balance Sheet line items "
            "(Equity Capital, Reserves, Borrowings, Fixed Assets, Total "
            "Assets, etc.) scraped from screener.in. One row per "
            "(company, line item, year)."
        ),
    },
    "statement_cash_flow": {
        "statement_key": "cash_flow",
        "description": (
            "Annual consolidated Cash Flow statement line items "
            "(Cash from Operating/Investing/Financing Activity, Net Cash "
            "Flow, Free Cash Flow, etc.) scraped from screener.in. One row "
            "per (company, line item, year)."
        ),
    },
}

# Shared column layout for every statement_* table.
STATEMENT_COLUMNS = [
    ("id", "INTEGER", "Surrogate primary key."),
    ("company_id", "INTEGER", "Foreign key -> companies.id"),
    ("line_item", "TEXT", "Row label exactly as shown on screener.in, e.g. 'Sales', 'Net Profit'."),
    ("period_label", "TEXT", "Period label as shown on screener.in, e.g. 'Mar 2024' (financial year ending March 2024)."),
    ("fiscal_year_end", "TEXT", "ISO date of the period end, e.g. '2024-03-31'. Derived from period_label."),
    ("unit", "TEXT", "Unit of `value` for this specific row, e.g. 'INR_CRORE', 'INR', 'PERCENT', 'RATIO'. Always check this column per-row rather than assuming a table-wide unit. See UNIT_LABELS in db.py for the full code -> human-readable mapping."),
    ("scraped_at", "TEXT", "UTC timestamp this row was last (re)scraped."),
]

# Canonical unit codes stored in statement_*.unit, and what they mean.
# screener.in publishes consolidated statement figures in Rs. Crores by
# default, but a handful of line items break that pattern (EPS is in plain
# Rupees; several rows are percentages or dimensionless ratios). The unit
# is therefore classified per ROW (see classify_unit() in ingest.py), not
# fixed for the whole table.
UNIT_LABELS = {
    "INR_CRORE": "Indian Rupees, in Crores (1 Crore = 10,000,000 / 1e7). "
                  "This is screener.in's default reporting unit for monetary line items.",
    "INR": "Indian Rupees, absolute value (not crores). Used for per-share figures like EPS.",
    "PERCENT": "A percentage value, e.g. 18 means 18%. Not a currency amount.",
    "RATIO": "A dimensionless ratio (e.g. CFO/OP). Not a currency amount.",
}

COMPANIES_COLUMNS = [
    ("id", "INTEGER", "Surrogate primary key."),
    ("input_symbol", "TEXT", "Symbol exactly as supplied by the caller at scrape time."),
    ("screener_symbol", "TEXT", "Symbol used in the screener.in URL (unique)."),
    ("company_name", "TEXT", "Full company name as shown on screener.in."),
    ("nse_symbol", "TEXT", "NSE trading symbol, if listed on NSE."),
    ("bse_code", "TEXT", "BSE numeric scrip code, if listed on BSE."),
    ("source_url", "TEXT", "screener.in URL the data was scraped from."),
    ("last_scraped_at", "TEXT", "UTC timestamp of the most recent successful scrape for this company."),
]


@contextmanager
def get_connection(db_path=None):
    """Context-managed sqlite3 connection with sane defaults.

    Row factory is sqlite3.Row so callers can access columns by name
    (row["value"]) as well as by index - this matters for the discovery
    API, which returns column-name-keyed dicts to be agent-friendly.
    """
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path=None):
    """Create all typed tables and metadata/discovery tables if they don't exist.

    Safe to call repeatedly (idempotent - uses CREATE TABLE IF NOT EXISTS).
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        # ---------------- typed table: companies ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_symbol TEXT NOT NULL,
                screener_symbol TEXT NOT NULL UNIQUE,
                company_name TEXT NOT NULL,
                nse_symbol TEXT,
                bse_code TEXT,
                source_url TEXT,
                last_scraped_at TEXT
            )
        """)

        # ---------------- typed tables: statements ----------------
        for table_name in STATEMENT_TABLES:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                    line_item TEXT NOT NULL,
                    period_label TEXT NOT NULL,
                    fiscal_year_end TEXT,
                    value REAL,
                    unit TEXT,
                    scraped_at TEXT NOT NULL,
                    UNIQUE(company_id, line_item, period_label)
                )
            """)
            # If this DB was created before the `unit` column existed,
            # add it now so existing installs upgrade in place instead of
            # silently missing the column (SQLite ignores the duplicate
            # column in CREATE TABLE IF NOT EXISTS above on existing DBs).
            existing_cols = {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()}
            if "unit" not in existing_cols:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN unit TEXT")
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_company
                ON {table_name}(company_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_line_item
                ON {table_name}(line_item)
            """)

        # ---------------- metadata/discovery tables ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _meta_tables (
                table_name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _meta_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                description TEXT NOT NULL,
                PRIMARY KEY (table_name, column_name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _meta_line_items (
                table_name TEXT NOT NULL,
                line_item TEXT NOT NULL,
                PRIMARY KEY (table_name, line_item)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _meta_units (
                unit_code TEXT PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)

        conn.commit()
        _seed_metadata(conn)


def _seed_metadata(conn):
    """(Re)populate _meta_tables and _meta_columns from the schema description
    constants above. Uses INSERT OR REPLACE so descriptions can be edited in
    this file and will sync on next init_schema() call.
    """
    cur = conn.cursor()

    cur.execute(
        "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
        ("companies", "master_data",
         "One row per company that has been scraped. Identity/lookup table - "
         "join statement_* tables to this on company_id."),
    )
    for col_name, data_type, desc in COMPANIES_COLUMNS:
        cur.execute(
            "INSERT OR REPLACE INTO _meta_columns (table_name, column_name, data_type, description) VALUES (?,?,?,?)",
            ("companies", col_name, data_type, desc),
        )

    for table_name, meta in STATEMENT_TABLES.items():
        cur.execute(
            "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
            (table_name, "financial_statement", meta["description"]),
        )
        for col_name, data_type, desc in STATEMENT_COLUMNS:
            cur.execute(
                "INSERT OR REPLACE INTO _meta_columns (table_name, column_name, data_type, description) VALUES (?,?,?,?)",
                (table_name, col_name, data_type, desc),
            )

    cur.execute(
        "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
        ("_meta_tables", "discovery",
         "Catalogue of every logical table in this database, with a "
         "description of what it holds. Start here to discover the schema."),
    )
    cur.execute(
        "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
        ("_meta_columns", "discovery",
         "Catalogue of every column in every logical table, with its type "
         "and meaning. Use after _meta_tables to learn a table's shape."),
    )
    cur.execute(
        "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
        ("_meta_line_items", "discovery",
         "Catalogue of every distinct line_item value present in each "
         "statement_* table, e.g. 'Sales', 'Net Profit'. Use to discover "
         "*what data* exists before filtering a query by line_item."),
    )
    cur.execute(
        "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
        ("_meta_units", "discovery",
         "Lookup table explaining every unit_code that can appear in the "
         "statement_*.unit column (e.g. INR_CRORE, INR, PERCENT, RATIO). "
         "Always join/look up a row's unit here rather than assuming a "
         "fixed unit for the whole table - EPS and %-rows differ from the "
         "Rs. Crores default."),
    )
    for unit_code, desc in UNIT_LABELS.items():
        cur.execute(
            "INSERT OR REPLACE INTO _meta_units (unit_code, description) VALUES (?,?)",
            (unit_code, desc),
        )

    conn.commit()


def refresh_line_item_catalogue(conn):
    """Rebuild _meta_line_items from whatever line_item values currently exist
    in the statement tables. Call this after every ingest so the discovery
    catalogue always reflects the actual data on disk.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM _meta_line_items")
    for table_name in STATEMENT_TABLES:
        cur.execute(f"SELECT DISTINCT line_item FROM {table_name}")
        for (line_item,) in cur.fetchall():
            cur.execute(
                "INSERT OR IGNORE INTO _meta_line_items (table_name, line_item) VALUES (?,?)",
                (table_name, line_item),
            )
    conn.commit()
