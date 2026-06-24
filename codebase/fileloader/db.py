
from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from datetime import datetime

from codebase.fileloader.schemas import (
    TABLE_NAME,
    CREATE_TABLE_SQL,
    ALL_INDEX_STATEMENTS,
    INSERTABLE_COLUMNS,
    SCRIP_PATTERN,
    YEAR_PATTERN,
    FILETYPE_PATTERN,
    FILENAME_PATTERN,
    ALLOWED_STATUS_VALUES,
    MAX_FILENAME_LENGTH,
    MAX_REASON_LENGTH,
    MAX_PATH_LENGTH,
    FORBIDDEN_CHARS_PATTERN,
    PATH_TRAVERSAL_PATTERN,
    ALLOWED_TABLES
)

from config import CONFIG
from codebase.common.exceptions import DatabaseValidationError
from codebase.fileloader.skelton import UploadResult

DB_PATH = Path(CONFIG.PROCESSED_FILES_DB_PATH)

# Single lock guarding all DB access made through this module.
_db_lock = threading.Lock()



# --------------------------------------------------------------------------
# Connection handling
# --------------------------------------------------------------------------

@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # WAL improves concurrent read/write behavior; busy_timeout makes
        # SQLite wait (instead of erroring immediately) if briefly locked.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> Path:
    CONFIG.PROCESSED_FILES_DB_PATH.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        with _get_connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for stmt in ALL_INDEX_STATEMENTS:
                conn.execute(stmt)
    return DB_PATH


# Field validation — defense in depth, run again right before insert.

def _check_forbidden_chars(field: str, value: str) -> None:
    if FORBIDDEN_CHARS_PATTERN.search(value):
        raise DatabaseValidationError(
            field, value, "contains forbidden control characters (NUL/CR/LF)."
        )


def _validate_record_fields(record: UploadResult) -> None:
    filename = record.filename
    scrip = record.scrip
    year = record.year
    filetype = record.file_type
    status = record.status
    reason = record.reason
    destination_path = record.destination_path
    upload_date = record.date
    upload_time = record.time

    # filename
    if not filename or not isinstance(filename, str):
        raise DatabaseValidationError("filename", filename, "must be a non-empty string.")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise DatabaseValidationError("filename", filename, "exceeds max length.")
    _check_forbidden_chars("filename", filename)
    if not FILENAME_PATTERN.match(filename):
        raise DatabaseValidationError(
            "filename", filename, "does not match required Scrip_Year_pdf.pdf pattern."
        )

    if scrip is not None:
        if not isinstance(scrip, str) or not SCRIP_PATTERN.match(scrip):
            raise DatabaseValidationError("scrip", scrip, "must be alphanumeric.")
        _check_forbidden_chars("scrip", scrip)

    # year
    if year is not None:
        if not isinstance(year, str) or not YEAR_PATTERN.match(year):
            raise DatabaseValidationError("year", year, "must be a 4-digit string.")

    # filetype
    if filetype is not None:
        if not isinstance(filetype, str) or not FILETYPE_PATTERN.match(filetype):
            raise DatabaseValidationError("filetype", filetype, "must be 'pdf'.")

    # status
    if status not in ALLOWED_STATUS_VALUES:
        raise DatabaseValidationError(
            "status", status, f"must be one of {ALLOWED_STATUS_VALUES}."
        )

    # reason
    if reason is not None:
        if not isinstance(reason, str):
            raise DatabaseValidationError("reason", reason, "must be a string.")
        if len(reason) > MAX_REASON_LENGTH:
            raise DatabaseValidationError("reason", reason, "exceeds max length.")
        _check_forbidden_chars("reason", reason)

    # destination_path (nullable)
    if destination_path is not None:
        if not isinstance(destination_path, str):
            raise DatabaseValidationError(
                "destination_path", destination_path, "must be a string."
            )
        if len(destination_path) > MAX_PATH_LENGTH:
            raise DatabaseValidationError(
                "destination_path", destination_path, "exceeds max length."
            )
        _check_forbidden_chars("destination_path", destination_path)
        if PATH_TRAVERSAL_PATTERN.search(destination_path):
            raise DatabaseValidationError(
                "destination_path", destination_path, "contains path traversal sequence."
            )

    # upload_date
    if not upload_date or not isinstance(upload_date, str):
        raise DatabaseValidationError(
            "upload_date", upload_date, "must be a non-empty string."
        )
    _check_forbidden_chars("upload_datetime", upload_date)

    #upload_time
    if not upload_time or not isinstance(upload_time, str):
        raise DatabaseValidationError(
            "upload_time", upload_time, "must be a non-empty string"
        )

#validate table name
def _validate_table_name(TABLE_NAME):
    if TABLE_NAME not in ALLOWED_TABLES:
        raise DatabaseValidationError(
            "Table Name",
            TABLE_NAME,
            "Table Name not in allowed Table Name list."
        )
    
def _validate_date_string(value: str, field: str) -> None:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            datetime.strptime(value, fmt)
            return
        except ValueError:
            continue
    raise DatabaseValidationError(field, value, "invalid date format.")

# Insert
def insert_upload_record(record: UploadResult) -> int:
    
    _validate_record_fields(record)
    _validate_table_name(TABLE_NAME)

    columns = INSERTABLE_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    values = tuple(getattr(record, col) for col in columns)


    sql = f"INSERT INTO {TABLE_NAME} ({column_list}) VALUES ({placeholders});"

    with _db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(sql, values)
            return cursor.lastrowid


# Queries (all parameterized — no string-formatted SQL, ever)

def get_record_by_filename(filename: str) -> dict[str, Any] | None:
    """Fetch the most recent record matching an exact filename, or None."""
    _validate_table_name(TABLE_NAME)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE filename = ? ORDER BY id DESC LIMIT 1;"
    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(sql, (filename,)).fetchone()
    return dict(row) if row else None


def get_records_by_scrip(scrip: str) -> list[dict[str, Any]]:
    """Fetch all records for a given scrip, most recent first."""
    _validate_table_name(TABLE_NAME)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE scrip = ? ORDER BY id DESC;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (scrip,)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_status(status: str) -> list[dict[str, Any]]:
    """Fetch all records with a given status ('SUCCESS' or 'FAILED')."""
    if status not in ALLOWED_STATUS_VALUES:
        raise DatabaseValidationError(
            "status", status, f"must be one of {ALLOWED_STATUS_VALUES}."
        )
    _validate_table_name(TABLE_NAME)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? ORDER BY id DESC;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (status,)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_date_range(start_date: str, end_date: str) -> list[dict[str, Any]]:

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}:\d{2})?$")

    if not date_pattern.match(start_date) or not date_pattern.match(end_date):
        raise DatabaseValidationError(
            "date_range", (start_date, end_date), "dates must be in YYYY-MM-DD format."
        )

    _validate_table_name(TABLE_NAME)
    sql = (
        f"SELECT * FROM {TABLE_NAME} "
        f"WHERE upload_datetime BETWEEN ? AND ? "
        f"ORDER BY id DESC;"
    )
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (start_date, end_date)).fetchall()
    return [dict(r) for r in rows]


def get_all_records() -> list[dict[str, Any]]:
    """Fetch all records, most recent first."""
    _validate_table_name(TABLE_NAME)
    sql = f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]