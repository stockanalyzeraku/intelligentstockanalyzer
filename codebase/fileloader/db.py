#YET TO ADD
#RETRY CONNECTION AFTER SOME TIME AND RELEASE RESOURCES
#PROGRESS BAR
"""
Database layer for the fileloader module.

Owns the SQLite connection, table/index creation, the insert of an upload
record, and all read queries against the processed_files table. No file
validation logic lives here — that's all in validator.py.
"""
from __future__ import annotations

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
    DATABASE_NAME,
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
    _validate_time,
)

from config import CONFIG
from codebase.fileloader.exceptions import DatabaseValidationError, FilenameValidationError
from codebase.fileloader.skelton import UploadResult


_db_lock = threading.Lock()


def _db_path() -> Path:
    """Return the folder where the SQLite database file lives."""
    return Path(CONFIG.PROCESSED_FILES_DB_PATH)


def _friendly_field_message(field_name: str) -> str:
    """Turn an internal field name into a short, plain-English error message."""
    return f"Invalid {field_name.replace('_', ' ')}. Could not save the record."


# Connection handling
@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with safe defaults, and close it when done."""
    db_path = f"{str(_db_path())}/{DATABASE_NAME}"
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # busy_timeout makes SQLite wait (instead of erroring immediately)
        # if briefly locked by another connection.
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
    """Create the processed_files table and its indexes if they don't exist yet."""
    if TABLE_NAME not in ALLOWED_TABLES:
        raise DatabaseValidationError(
            "Table Name",
            TABLE_NAME,
            "Table Name not in allowed Table Name list.",
        )
    _db_path().mkdir(parents=True, exist_ok=True)
    with _db_lock:
        with _get_connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for stmt in ALL_INDEX_STATEMENTS:
                conn.execute(stmt)
    return _db_path()


def _run_field_validations(record: UploadResult, logger: StructuredLogger) -> str | None:
    """
    Check every field on an upload record before it is saved.
    Logs each field as it passes or fails. Returns a friendly error message
    if a field fails, or None if everything is valid.
    """
    validations = (
        ("filename", lambda: _validate_filename(record.filename)),
        ("scrip", lambda: _validate_scrip(record.scrip)),
        ("year", lambda: _validate_year(record.year)),
        ("filetype", lambda: _validate_filetype(record.file_type)),
        ("status", lambda: _validate_status(record.status)),
        ("reason", lambda: _validate_reason(record.reason)),
        ("destination_path", lambda: _validate_destination_path(record.destination_path)),
        ("upload_date", lambda: _validate_parse_date(record.date, "upload_date")),
        ("upload_time", lambda: _validate_time(record.time)),
    )

    for field_name, check in validations:
        try:
            check()
        except (DatabaseValidationError, FilenameValidationError) as exc:
            logger.event(
                f"{record.filename} : Validation failed on field '{field_name}': {exc}",
                filename=record.filename, step="validation", field=field_name,
                outcome="failed",
            )
            return _friendly_field_message(field_name)

    logger.event(
        f"{record.filename} : All fields validated successfully",
        filename=record.filename, step="validation", outcome="passed",
    )
    return None


# Insert
def insert_upload_record(record: UploadResult, logger: StructuredLogger) -> int | str:
    """
    Validate an upload record and save it to the database.
    Returns the new row's id on success, or a short error message on failure.
    """
    logger.event(
        f"{record.filename} : Validation started for DB insert",
        filename=record.filename, step="validation", stage="start",
    )

    failure = _run_field_validations(record, logger)
    if failure is not None:
        return failure

    columns = INSERTABLE_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    values = tuple(getattr(record, col) for col in columns)
    sql = f"INSERT INTO {TABLE_NAME} ({column_list}) VALUES ({placeholders});"

    logger.event(
        f"{record.filename} : Inserting record into '{TABLE_NAME}'",
        filename=record.filename, step="db_insert", stage="start",
    )

    try:
        with _db_lock:
            with _get_connection() as conn:
                cursor = conn.execute(sql, values)
                rowid = cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error(
            f"{record.filename} : Database error while inserting record: {exc}",
            filename=record.filename, step="db_insert", outcome="failed",
            exception_type=type(exc).__name__,
        )
        raise

    if rowid is None:
        logger.error(
            f"{record.filename} : INSERT returned no row id — insert may have silently failed",
            filename=record.filename, step="db_insert", outcome="failed",
        )
        raise RuntimeError("INSERT returned no row ID — insert may have silently failed.")

    logger.info(
        f"{record.filename} : Record inserted successfully (row id {rowid})",
        filename=record.filename, step="db_insert", outcome="passed", row_id=rowid,
    )
    return rowid


# Queries (all parameterized — no string-formatted SQL, ever)

def get_record_by_filename(filename: str, logger: StructuredLogger) -> dict[str, Any] | None:
    """Look up the most recent record for an exact filename. Returns None if not found."""
    logger.event(
        f"{filename} : Looking up record by filename",
        filename=filename, step="get_record_by_filename", stage="start",
    )
    _validate_filename(filename)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE filename = ? ORDER BY id DESC LIMIT 1;"
    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(sql, (filename,)).fetchone()
    found = row is not None
    logger.event(
        f"{filename} : Lookup by filename {'found a record' if found else 'found no record'}",
        filename=filename, step="get_record_by_filename", outcome="passed", found=found,
    )
    return dict(row) if row else None


def get_records_by_scrip(scrip: str, logger: StructuredLogger, limit: int = 1000) -> list[dict[str, Any]]:
    """Look up all records for a given scrip (stock symbol), newest first."""
    logger.event(
        f"Looking up records for scrip '{scrip}'",
        scrip=scrip, step="get_records_by_scrip", stage="start",
    )
    _validate_limit(limit)
    _validate_scrip(scrip)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE scrip = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (scrip, limit)).fetchall()
    logger.event(
        f"Found {len(rows)} record(s) for scrip '{scrip}'",
        scrip=scrip, step="get_records_by_scrip", outcome="passed", count=len(rows),
    )
    return [dict(r) for r in rows]


def get_records_by_status(status: str, logger: StructuredLogger, limit: int = 1000) -> list[dict[str, Any]]:
    """Look up all records with a given status (SUCCESS or FAILED), newest first."""
    logger.event(
        f"Looking up records with status '{status}'",
        status=status, step="get_records_by_status", stage="start",
    )
    _validate_limit(limit)
    _validate_status(status)

    sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (status, limit)).fetchall()
    logger.event(
        f"Found {len(rows)} record(s) with status '{status}'",
        status=status, step="get_records_by_status", outcome="passed", count=len(rows),
    )
    return [dict(r) for r in rows]


def get_records_by_date_range(
    start_date: str, end_date: str, logger: StructuredLogger, limit: int = 1000
) -> list[dict[str, Any]]:
    """Look up all records uploaded between two dates (inclusive), newest first."""
    logger.event(
        f"Looking up records between '{start_date}' and '{end_date}'",
        start_date=start_date, end_date=end_date,
        step="get_records_by_date_range", stage="start",
    )
    _validate_limit(limit)
    start_dt = _validate_parse_date(start_date, "start_date")
    end_dt = _validate_parse_date(end_date, "end_date")
    if start_dt > end_dt:
        logger.event(
            f"Invalid date range: start '{start_date}' is after end '{end_date}'",
            start_date=start_date, end_date=end_date,
            step="get_records_by_date_range", outcome="failed",
        )
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
    logger.event(
        f"Found {len(rows)} record(s) between '{start_date}' and '{end_date}'",
        start_date=start_date, end_date=end_date,
        step="get_records_by_date_range", outcome="passed", count=len(rows),
    )
    return [dict(r) for r in rows]


def get_all_records(logger: StructuredLogger, limit: int = 1000) -> list[dict[str, Any]]:
    """Look up every record in the table, newest first, up to `limit` rows."""
    logger.event(
        "Looking up all records",
        step="get_all_records", stage="start", limit=limit,
    )
    _validate_limit(limit)
    sql = f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
    logger.event(
        f"Found {len(rows)} record(s) in total",
        step="get_all_records", outcome="passed", count=len(rows),
    )
    return [dict(r) for r in rows]