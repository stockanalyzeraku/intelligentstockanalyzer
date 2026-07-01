"""Database layer for the cleaning pipeline.

Single responsibility: own the SQLite connection and the two operations
the pipeline needs — insert a processed-file record, look one up by filename.

No validation logic lives here (that belongs in validator.py).
No schema SQL lives here (that belongs in schemas.py).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from config import CONFIG
from codebase.cleaning.validator import validate_db_insert_record
from codebase.cleaning.schemas import (
    DATABASE_NAME,
    TABLE_NAME,
    INSERT_SQL,
    SELECT_BY_FILENAME_SQL,
    SELECT_BY_SCRIP_SQL,
)


_db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    return Path(CONFIG.PROCESSED_FILES_DB_PATH)


@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with safe defaults; commit on exit, rollback on error."""
    conn = sqlite3.connect(
        str(_db_path() / DATABASE_NAME),
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA trusted_schema=OFF;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_cleaning_record(
    filename:       str,
    scrip:          str,
    year:           int,
    cleaned_path:   str,
    embedding_path: str,
    status:         str = "SUCCESS",
    reason:         str = "",
) -> int:
    """Insert one record for a completed pipeline run.

    Maps the cleaning-specific fields onto the shared processed_files table:
      destination_path → cleaned_path
      reason           → embedding_path  (when status is SUCCESS)

    Returns the new row id.
    """
    # Validate every field before touching the database.
    filename, scrip, year, cleaned_path, embedding_path, status, reason = validate_db_insert_record(
        filename       = filename,
        scrip          = scrip,
        year           = year,
        cleaned_path   = cleaned_path,
        embedding_path = embedding_path,
        status         = status,
        reason         = reason,
    )

    now   = datetime.now()
    date  = now.strftime("%Y-%m-%d")
    time  = now.strftime("%H:%M:%S")

    # When the run succeeded, reason holds the embedding path so both output
    # paths survive in the existing schema without altering the table.
    db_reason = embedding_path if status == "SUCCESS" else reason

    values = (
        filename,
        scrip,
        str(year),
        "json",          # file_type is always json at this pipeline stage
        status,
        db_reason,
        cleaned_path,    # destination_path
        date,
        time,
    )

    with _db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(INSERT_SQL, values)
            row_id = cursor.lastrowid

    if row_id is None:
        raise RuntimeError("INSERT returned no row id — insert may have silently failed.")

    return row_id


def get_record_by_filename(filename: str) -> dict[str, Any] | None:
    """Return the most recent record for a filename, or None if not found."""
    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(SELECT_BY_FILENAME_SQL, (filename,)).fetchone()
    return dict(row) if row else None


def get_records_by_scrip(scrip: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Return all records for a scrip, newest first."""
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(SELECT_BY_SCRIP_SQL, (scrip, limit)).fetchall()
    return [dict(r) for r in rows]