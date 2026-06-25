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
    ALLOWED_TABLES,
    TIME_PATTERN,
    DATE_PATTERN
)

from config import CONFIG
from codebase.common.exceptions import DatabaseValidationError
from codebase.fileloader.skelton import UploadResult


_db_lock = threading.Lock()
def _db_path() -> Path:
    return Path(CONFIG.PROCESSED_FILES_DB_PATH)

# Connection handling
@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:

    conn = sqlite3.connect(str(_db_path()), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # WAL improves concurrent read/write behavior; busy_timeout makes
        # SQLite wait (instead of erroring immediately) if briefly locked.
        row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
        if row[0].lower() != "wal":
            raise RuntimeError("failed to sel WAL Journal Mode")
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
    with _db_lock:
        with _get_connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for stmt in ALL_INDEX_STATEMENTS:
                conn.execute(stmt)
    return _db_path()


# Field validation — defense in depth, run again right before insert.
def _check_forbidden_chars(field: str, value: str) -> None:
    if FORBIDDEN_CHARS_PATTERN.search(value):
        raise DatabaseValidationError(
            field, value, "contains forbidden control characters (NUL/CR/LF)."
        )


#validate table name
if TABLE_NAME not in ALLOWED_TABLES:
    raise DatabaseValidationError(
        "Table Name",
        TABLE_NAME,
        "Table Name not in allowed Table Name list."
        )
    
#validate date    
def _validate_date_string(value: str, field: str) -> None:
    for fmt in DATE_PATTERN:
        try:
            datetime.strptime(value, fmt)
            return
        except ValueError:
            continue
    raise DatabaseValidationError(field, value, "invalid date format.")

#validate filename
def _validate_filename(filename:str) -> None:
    if not filename or not isinstance(filename, str):
        raise DatabaseValidationError("filename", filename, "must be a non-empty string.")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise DatabaseValidationError("filename", filename, "exceeds max length.")
    _check_forbidden_chars("filename", filename)
    if not FILENAME_PATTERN.match(filename):
        raise DatabaseValidationError(
            "filename", filename, "does not match required Scrip_Year_pdf.pdf pattern."
        )
 
#validate scrip
def _validate_scrip(scrip:str | None) -> None:
    if scrip is not None:
        if not isinstance(scrip, str) or not SCRIP_PATTERN.match(scrip):
            raise DatabaseValidationError("scrip", scrip, "must be alphanumeric.")
        _check_forbidden_chars("scrip", scrip)
    else:
        raise DatabaseValidationError("scrip", scrip, "scrip is empty")

#validate all records
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
    _validate_filename(filename)

    #script
    _validate_scrip(scrip)

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
    _validate_date_string(upload_date, "upload_date")
    _check_forbidden_chars("upload_date", upload_date)

    #upload_time
    if not upload_time or not isinstance(upload_time, str) or not TIME_PATTERN.match(upload_time):
        raise DatabaseValidationError(
            "upload_time", upload_time, "must be a non-empty string"
        )
    _check_forbidden_chars("upload_time", upload_time)
    
#limit records
def _validate_limit(limit:int = 1000)-> None:
    if not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if not (1 <= limit <= 1000):
        raise ValueError("limit must be between 1 and 1000.")

#parse date
def _parse_date(value: str, field: str) -> datetime:
    for fmt in DATE_PATTERN:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise DatabaseValidationError(field, value, "invalid date format.")
        

# Insert
def insert_upload_record(record: UploadResult) -> int:
    
    _validate_record_fields(record)
    columns = INSERTABLE_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    values = tuple(getattr(record, col) for col in columns)


    sql = f"INSERT INTO {TABLE_NAME} ({column_list}) VALUES ({placeholders});"

    with _db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(sql, values)
            rowid = cursor.lastrowid
            if rowid is None:
                raise RuntimeError("INSERT returned no row ID — insert may have silently failed.")
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
    if status not in ALLOWED_STATUS_VALUES:
        raise DatabaseValidationError(
            "status", status, f"must be one of {ALLOWED_STATUS_VALUES}."
        )
    sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (status,limit)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_date_range(start_date: str, end_date: str, limit: int = 1000) -> list[dict[str, Any]]:

    _validate_limit(limit)
    _validate_date_string(start_date, "start_date")
    _validate_date_string(end_date, "end_date")
    start_dt = _parse_date(start_date, "start_date")
    end_dt   = _parse_date(end_date, "end_date")
    if start_dt > end_dt:
        raise DatabaseValidationError(
            "date_range", (start_date, end_date),
            "start_date must be before or equal to end_date."
    )
    sql = (
        f"SELECT * FROM {TABLE_NAME} "
        f"WHERE upload_datetime BETWEEN ? AND ? "
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