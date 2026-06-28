#YET TO ADD
#RETRY CONNECTION AFTER SOME TIME AND RELEASE RESOURCES
#PROGRESS BAR
from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from logger import StructuredLogger

from codebase.fileloader.schemas import (
    TABLE_NAME,
    CREATE_TABLE_SQL,
    ALL_INDEX_STATEMENTS,
    INSERTABLE_COLUMNS,
    ALLOWED_TABLES,
    DATABASE_NAME
)

from codebase.fileloader.validator import (
    _validate_filename,
    _validate_limit,
    _validate_parse_date,
    _validate_scrip,
    _validate_status,
    _validate_year,
    _validate_filetype,
    _validate_destination_path,
    _validate_reason,
    _validate_time
)

from config import CONFIG
from codebase.fileloader.exceptions import DatabaseValidationError
from codebase.fileloader.skelton import UploadResult


_db_lock = threading.Lock()
def _db_path() -> Path:
    return Path(CONFIG.PROCESSED_FILES_DB_PATH)

# Connection handling
@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    db_path = f"{str(_db_path())}/{DATABASE_NAME}"
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # WAL improves concurrent read/write behavior; busy_timeout makes
        # SQLite wait (instead of erroring immediately) if briefly locked.
        # row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
        # if row[0].lower() != "wal":
        #     raise RuntimeError("failed to sel WAL Journal Mode")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> Path:
    _db_path().mkdir(parents=True, exist_ok=True)
    #validate table name
    if TABLE_NAME not in ALLOWED_TABLES:
        raise DatabaseValidationError(
            "Table Name",
            TABLE_NAME,
            "Table Name not in allowed Table Name list."
            )
    with _db_lock:
        with _get_connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for stmt in ALL_INDEX_STATEMENTS:
                conn.execute(stmt)
    return _db_path()

    
# Insert
def insert_upload_record(record: UploadResult, logger: StructuredLogger) -> int | str:
    logger.event(f"Validation Started for record upload to DB")
    # filename
    try:
        _validate_filename(record.filename)
    except:
        logger.event(f"Filename is not valid")
        return "Filename is not valid"
    #script
    _validate_scrip(record.scrip)
    # year
    _validate_year(record.year)
    # filetype
    _validate_filetype(record.file_type)
    # status
    _validate_status(record.status)
    # reason
    _validate_reason(record.status)
    # destination_path (nullable)
    _validate_destination_path(record.destination_path)
    # upload_date
    _validate_parse_date(record.date, "upload_date")
    #upload_time
    _validate_time(record.time)

    columns = INSERTABLE_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    values = tuple(getattr(record, col) for col in columns)


    sql = f"INSERT INTO {TABLE_NAME} ({column_list}) VALUES ({placeholders});"

    logger.event("Upload to db started")
    with _db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(sql, values)
            rowid = cursor.lastrowid
            if rowid is None:
                logger.event("Record was not updated in db")
                raise RuntimeError("INSERT returned no row ID — insert may have silently failed.")
            logger.event("Record was updated in db")
            return rowid


# Queries (all parameterized — no string-formatted SQL, ever)

def get_record_by_filename(filename: str) -> dict[str, Any] | None:
    """Fetch the most recent record matching an exact filename, or None."""
    _validate_filename(filename)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE filename = ? ORDER BY id DESC LIMIT 1;"
    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(sql, (filename,)).fetchone()
    return dict(row) if row else None


def get_records_by_scrip(scrip: str, limit:int = 1000) -> list[dict[str, Any]]:
    _validate_limit(limit)
    _validate_scrip(scrip)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE scrip = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (scrip,limit)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_status(status: str, limit:int = 1000) -> list[dict[str, Any]]:
    _validate_limit(limit) 
    _validate_status(status)

    sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (status,limit)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_date_range(start_date: str, end_date: str, limit: int = 1000) -> list[dict[str, Any]]:

    _validate_limit(limit)
    start_dt = _validate_parse_date(start_date, "start_date")
    end_dt   = _validate_parse_date(end_date, "end_date")
    if start_dt > end_dt:
        raise DatabaseValidationError(
            "date_range", (start_date, end_date),
            "start_date must be before or equal to end_date."
    )
    sql = (
        f"SELECT * FROM {TABLE_NAME} "
        f"WHERE date BETWEEN ? AND ? "
        f"ORDER BY id DESC LIMIT ?;"
    )
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (start_date, end_date, limit)).fetchall()
    return [dict(r) for r in rows]


def get_all_records(limit:int = 1000) -> list[dict[str, Any]]:
    """Fetch all records, most recent first."""
    _validate_limit(limit)
    sql = f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]