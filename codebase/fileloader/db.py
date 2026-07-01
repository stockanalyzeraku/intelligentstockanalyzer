#YET TO ADD
#RETRY CONNECTION AFTER SOME TIME AND RELEASE RESOURCES
#PROGRESS BAR
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
    _validate_ocr_status,
    _validate_ocr_reason,
)

from config import CONFIG
from codebase.fileloader.exceptions import (
    DatabaseValidationError,
    FilenameValidationError,
    RecordNotFoundError,
)
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
    Run every per-field validator against `record`, logging each outcome.
    Returns a short failure string on the first failed field, or None if
    every field passed.
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
            return f"{field_name} is not valid"

    logger.event(
        f"{record.filename} : All fields validated successfully",
        filename=record.filename, step="validation", outcome="passed",
    )
    return None


# Insert
def insert_upload_record(record: UploadResult, logger: StructuredLogger) -> int | str:
    """
    Validate every field on `record` and persist it to the processed-files
    table. Every validation step and the DB write itself are logged with
    enough context (filename, field, reason) to diagnose a failure from the
    logs alone, without needing to reproduce it.
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

def get_record_by_filename(filename: str) -> dict[str, Any] | None:
    """Fetch the most recent record matching an exact filename, or None."""
    _validate_filename(filename)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE filename = ? ORDER BY id DESC LIMIT 1;"
    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(sql, (filename,)).fetchone()
    return dict(row) if row else None


def get_records_by_scrip(scrip: str, limit: int = 1000) -> list[dict[str, Any]]:
    _validate_limit(limit)
    _validate_scrip(scrip)
    sql = f"SELECT * FROM {TABLE_NAME} WHERE scrip = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (scrip, limit)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_status(status: str, limit: int = 1000) -> list[dict[str, Any]]:
    _validate_limit(limit)
    _validate_status(status)

    sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (status, limit)).fetchall()
    return [dict(r) for r in rows]


def get_records_by_date_range(start_date: str, end_date: str, limit: int = 1000) -> list[dict[str, Any]]:

    _validate_limit(limit)
    start_dt = _validate_parse_date(start_date, "start_date")
    end_dt = _validate_parse_date(end_date, "end_date")
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


def get_all_records(limit: int = 1000) -> list[dict[str, Any]]:
    """Fetch all records, most recent first."""
    _validate_limit(limit)
    sql = f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT ?;"
    with _db_lock:
        with _get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


# Update — OCR pipeline result

def update_ocr_status(
    scrip: str,
    year: str,
    file_type: str,
    ocr_status: str,
    ocr_reason: str | None = None,
) -> int:
    """
    Update the ocr_status/ocr_reason columns on the row matching
    (scrip, year, file_type).

    The matching row is resolved to its exact `id` first (most recent
    match if duplicates exist), then updated by that id — this guarantees
    exactly one row changes even if scrip/year/file_type is not yet
    enforced as unique at the database level.

    Args:
        scrip:      Company/script identifier, as stored on the row.
        year:       4-digit year, as stored on the row.
        file_type:  File type ("pdf"), as stored on the row.
        ocr_status: One of "PENDING", "SUCCESS", "FAILED".
        ocr_reason: Optional detail — error message on failure, None/"" on success.

    Returns:
        The id of the row that was updated.

    Raises:
        DatabaseValidationError: if any input fails validation.
        RecordNotFoundError:     if no row matches (scrip, year, file_type).
    """
    _validate_scrip(scrip)
    _validate_year(year)
    _validate_filetype(file_type)
    _validate_ocr_status(ocr_status)
    _validate_ocr_reason(ocr_reason)

    select_sql = (
        f"SELECT id FROM {TABLE_NAME} "
        f"WHERE scrip = ? AND year = ? AND file_type = ? "
        f"ORDER BY id DESC LIMIT 1;"
    )
    update_sql = (
        f"UPDATE {TABLE_NAME} SET ocr_status = ?, ocr_reason = ? WHERE id = ?;"
    )

    with _db_lock:
        with _get_connection() as conn:
            row = conn.execute(select_sql, (scrip, year, file_type)).fetchone()
            if row is None:
                raise RecordNotFoundError(scrip, year, file_type)
            record_id = row["id"]
            conn.execute(update_sql, (ocr_status, ocr_reason, record_id))

    return record_id