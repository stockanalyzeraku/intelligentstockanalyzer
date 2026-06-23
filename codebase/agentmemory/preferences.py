"""SQLite-backed user preference memory.

New, additive module - does NOT modify cachememory.py, cachestructure.py,
dbstructure.py, or workingmemory.py. Mirrors CacheMemory's own
connect()/initialise_database() pattern for consistency within this
package, but is otherwise fully independent: a separate table
(user_preferences, from preferencestructure.py), a separate class, no
shared state with CacheMemory.

SCOPE (current, single-user system): every call defaults to a single
constant DEFAULT_USER_ID. This was a deliberate simplification - there is
no real multi-user/auth concept in this system yet. The user_id column
still exists (rather than hardcoding away the concept entirely) so this
can be extended to real per-user preferences later without a schema
migration - only the caller-supplied user_id would need to change.

SUPPORTED PREFERENCE KEYS (the agreed initial set - see PreferenceKeys):
    - trailing_years              : int, overrides clarification.py's
                                     DEFAULT_TRAILING_YEARS for this user.
    - always_include_qualitative  : bool, if true, Stage 2 should treat
                                     needs_qualitative_context as True
                                     even if Stage 1 didn't detect a "why"-
                                     style question.
    - preferred_unit              : str, e.g. "INR_CRORE" - a DISPLAY
                                     preference only. This does NOT change
                                     what unit values are stored or
                                     fetched in - financials.db always
                                     stores screener.in's native unit per
                                     row (see db.py UNIT_LABELS). This
                                     preference is read by the Synthesis
                                     prompt as a hint for how to phrase
                                     output, not a conversion instruction -
                                     no unit conversion math is performed
                                     by this system today.

A preference set here only ever supplies a DEFAULT. It is intentionally
designed to never override an explicit instruction in the user's own
query (e.g. "just give me 2023, nothing else" always wins over a stored
trailing_years preference) - that precedence is enforced in
clarification.py/pipeline.py, not here. This module only stores and
retrieves; it has no opinion on precedence.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from codebase.agentmemory.preferencestructure import (
    PREFERENCES_SCHEMA_STATEMENTS,
    USER_PREFERENCES_TABLE,
)

DEFAULT_USER_ID = "default_user"


class PreferenceKeys:
    """Recognized preference keys - a closed, documented set rather than
    arbitrary free-text keys, so a typo'd key doesn't silently do nothing.
    """

    TRAILING_YEARS = "trailing_years"
    ALWAYS_INCLUDE_QUALITATIVE = "always_include_qualitative"
    PREFERRED_UNIT = "preferred_unit"

    ALL = (TRAILING_YEARS, ALWAYS_INCLUDE_QUALITATIVE, PREFERRED_UNIT)


class UserPreferences:
    """Persist and retrieve simple key/value preferences per user."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        default_db_path = Path(__file__).resolve().parents[2] / "database" / "brain.db"
        self.db_path = Path(db_path or default_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise_database()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise_database(self) -> None:
        with self.connect() as conn:
            for statement in PREFERENCES_SCHEMA_STATEMENTS:
                conn.execute(statement)

    def set_preference(
        self, preference_key: str, value: Any, user_id: str = DEFAULT_USER_ID
    ) -> None:
        """Store a preference value (JSON-encoded, so bool/int/str/list/dict
        all round-trip correctly through get_preference).

        Raises ValueError if preference_key isn't in PreferenceKeys.ALL,
        to catch a typo'd key immediately rather than have it silently do
        nothing forever.
        """
        if preference_key not in PreferenceKeys.ALL:
            raise ValueError(
                f"Unknown preference key '{preference_key}'. Valid keys: {PreferenceKeys.ALL}"
            )
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {USER_PREFERENCES_TABLE} (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, preference_key) DO UPDATE SET
                    preference_value = excluded.preference_value,
                    updated_at = excluded.updated_at
                """,
                (user_id, preference_key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )

    def get_preference(
        self, preference_key: str, default: Any = None, user_id: str = DEFAULT_USER_ID
    ) -> Any:
        """Return a stored preference value, or `default` if unset."""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT preference_value FROM {USER_PREFERENCES_TABLE} WHERE user_id = ? AND preference_key = ?",
                (user_id, preference_key),
            ).fetchone()
            if row is None:
                return default
            return json.loads(row["preference_value"])

    def get_all_preferences(self, user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
        """Return every stored preference for a user as a plain dict."""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT preference_key, preference_value FROM {USER_PREFERENCES_TABLE} WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return {row["preference_key"]: json.loads(row["preference_value"]) for row in rows}

    def delete_preference(self, preference_key: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Remove a stored preference. Returns True if a row was deleted."""
        with self.connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {USER_PREFERENCES_TABLE} WHERE user_id = ? AND preference_key = ?",
                (user_id, preference_key),
            )
            return cursor.rowcount > 0
