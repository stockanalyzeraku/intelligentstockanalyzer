"""Learned/curated company name aliases.

Extends company resolution beyond the hardcoded COMPANY_LOOKUP dict in
codebase/agent/classify.py into a persisted, growable table:
    alias_text (lowercased) -> screener_symbol

This is entity-RESOLUTION data, not conversational memory, which is why it
lives in codebase/financials/ alongside companies/discovery rather than in
codebase/agentmemory/ - it answers "what company does this text refer to",
the same kind of question companies.screener_symbol already answers, just
for arbitrary user-typed phrases instead of canonical symbols.

WRITE POLICY (product decision): an alias is only auto-saved when Stage 1
self-reports confident=True for that resolution (see
codebase/agent/schemas.py QueryUnderstanding.confident and
codebase/agent/query_understanding.py). This is a soft signal (LLM
self-report, not a hard string match), so every write is logged with its
source and the exact query that produced it, and rows are trivially
deletable - a bad confident resolution is cheap to spot and remove, and
this table only ever affects a LOOKUP, never a verified financial number.

This module deliberately does NOT create its own separate database file -
it adds one table into the SAME financials.db (via the same
codebase.financials.db.get_connection helper), since aliases are
meaningless without the companies table they point into.
"""

from __future__ import annotations

from datetime import datetime, timezone

from codebase.financials import db

ALIASES_TABLE = "company_aliases"


def init_aliases_schema(db_path: str | None = None) -> None:
    """Create the company_aliases table if it doesn't exist, and register
    it in the existing _meta_tables/_meta_columns discovery tables so it's
    discoverable the same way every other table in financials.db is.

    Safe to call repeatedly. Call this alongside (after) db.init_schema().
    """
    db.init_schema(db_path)  # ensure companies + _meta_* tables exist first

    with db.get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {ALIASES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias_text TEXT NOT NULL UNIQUE,
                screener_symbol TEXT NOT NULL REFERENCES companies(screener_symbol),
                source TEXT NOT NULL,
                source_query TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{ALIASES_TABLE}_symbol
            ON {ALIASES_TABLE}(screener_symbol)
        """)

        cur.execute(
            "INSERT OR REPLACE INTO _meta_tables (table_name, category, description) VALUES (?,?,?)",
            (
                ALIASES_TABLE,
                "entity_resolution",
                "Learned and/or manually curated mapping of free-text company "
                "names/phrases to a screener_symbol, e.g. 'kalyan' -> 'KALYANKJIL'. "
                "Used to resolve a company before/alongside the LLM's own "
                "world knowledge. 'source' is either 'llm_confident' (auto-saved "
                "when Stage 1 self-reported high confidence) or 'manual' "
                "(curated by a human) - check this column before trusting an "
                "alias blindly.",
            ),
        )
        for col_name, data_type, desc in [
            ("id", "INTEGER", "Surrogate primary key."),
            ("alias_text", "TEXT", "Free-text phrase, lowercased, e.g. 'kalyan jewellers'. Unique."),
            ("screener_symbol", "TEXT", "The company this alias resolves to - foreign key to companies.screener_symbol."),
            ("source", "TEXT", "'llm_confident' (auto-learned) or 'manual' (human-curated)."),
            ("source_query", "TEXT", "The original user query that produced this alias, if auto-learned. NULL for manual entries."),
            ("created_at", "TEXT", "UTC timestamp this alias was added."),
        ]:
            cur.execute(
                "INSERT OR REPLACE INTO _meta_columns (table_name, column_name, data_type, description) VALUES (?,?,?,?)",
                (ALIASES_TABLE, col_name, data_type, desc),
            )
        conn.commit()


def resolve_alias(alias_text: str, db_path: str | None = None) -> str | None:
    """Look up alias_text (case-insensitive) and return its screener_symbol,
    or None if no alias is recorded for it.
    """
    normalized = alias_text.strip().lower()
    with db.get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT screener_symbol FROM {ALIASES_TABLE} WHERE alias_text = ?",
            (normalized,),
        ).fetchone()
        return row["screener_symbol"] if row else None


def save_alias(
    alias_text: str,
    screener_symbol: str,
    source: str = "llm_confident",
    source_query: str | None = None,
    db_path: str | None = None,
) -> bool:
    """Record a new alias, or silently no-op if one already exists for
    this exact alias_text (first write wins - does not overwrite an
    existing alias, manual or otherwise, with a new auto-learned one).

    Parameters
    ----------
    alias_text : str
        The free-text phrase to remember, e.g. the company_name_as_given
        from a confident QueryUnderstanding result.
    screener_symbol : str
        Must already exist in companies.screener_symbol (the table has a
        REFERENCES constraint, but SQLite does not enforce foreign keys by
        default - see note below).
    source : str
        'llm_confident' (default, for auto-learned aliases) or 'manual'.
    source_query : str, optional
        The original user query, for auditability of auto-learned aliases.

    Returns
    -------
    bool
        True if a new alias was written, False if one already existed for
        this alias_text (no-op) or the screener_symbol isn't a known company.

    Note: SQLite does not enforce FOREIGN KEY constraints unless
    "PRAGMA foreign_keys = ON" is set on the connection - db.get_connection
    does set this, so an invalid screener_symbol will raise
    sqlite3.IntegrityError here rather than silently inserting a dangling
    alias. This function catches that and returns False instead of raising,
    since a failed alias write should never break the calling pipeline.
    """
    normalized = alias_text.strip().lower()
    if not normalized:
        return False

    with db.get_connection(db_path) as conn:
        existing = conn.execute(
            f"SELECT id FROM {ALIASES_TABLE} WHERE alias_text = ?", (normalized,)
        ).fetchone()
        if existing is not None:
            return False

        try:
            conn.execute(
                f"""
                INSERT INTO {ALIASES_TABLE}
                    (alias_text, screener_symbol, source, source_query, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, screener_symbol, source, source_query, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return True
        except Exception:
            # Covers sqlite3.IntegrityError (unknown screener_symbol) and
            # any other unexpected DB error - alias-learning is a
            # best-effort convenience, never allowed to break the caller.
            return False


def list_aliases(screener_symbol: str | None = None, db_path: str | None = None) -> list[dict]:
    """List every alias, optionally filtered to one company. Useful for
    debugging/auditing what's been auto-learned.
    """
    with db.get_connection(db_path) as conn:
        if screener_symbol:
            rows = conn.execute(
                f"SELECT * FROM {ALIASES_TABLE} WHERE screener_symbol = ? ORDER BY created_at",
                (screener_symbol,),
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM {ALIASES_TABLE} ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def delete_alias(alias_text: str, db_path: str | None = None) -> bool:
    """Remove an alias by its exact text. Returns True if a row was deleted."""
    normalized = alias_text.strip().lower()
    with db.get_connection(db_path) as conn:
        cursor = conn.execute(f"DELETE FROM {ALIASES_TABLE} WHERE alias_text = ?", (normalized,))
        conn.commit()
        return cursor.rowcount > 0
