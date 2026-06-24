"""SQLite schema for user preference memory.

New, additive module - does NOT modify cachestructure.py, cachememory.py,
dbstructure.py, or workingmemory.py in any way. Mirrors the existing
schema-definition pattern used by cachestructure.py exactly, so this
slots into the same codebase.agentmemory package consistently.
"""

from __future__ import annotations
import sys
import os

USER_PREFERENCES_TABLE = "user_preferences"

PREFERENCES_SCHEMA_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS {USER_PREFERENCES_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        preference_key TEXT NOT NULL,
        preference_value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(user_id, preference_key)
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{USER_PREFERENCES_TABLE}_user ON {USER_PREFERENCES_TABLE}(user_id)",
)
