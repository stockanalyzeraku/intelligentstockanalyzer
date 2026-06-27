
from __future__ import annotations
from dataclasses import dataclass
from codebase.fileloader.skelton import UploadResult

import re

TABLE_NAME = "processed_files"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    filename          TEXT    NOT NULL,
    scrip             TEXT,
    year              TEXT,
    file_type         TEXT,
    status            TEXT    NOT NULL CHECK (status IN ('SUCCESS', 'FAILED')),
    reason            TEXT,
    destination_path  TEXT,
    upload_datetime   TEXT    NOT NULL,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# Helpful for filtering by scrip/status later, since this is a single shared table.
CREATE_INDEX_SCRIP_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_scrip ON {TABLE_NAME} (scrip);"
)
CREATE_INDEX_STATUS_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_status ON {TABLE_NAME} (status);"
)

ALL_INDEX_STATEMENTS = [CREATE_INDEX_SCRIP_SQL, CREATE_INDEX_STATUS_SQL]

# Columns allowed to be inserted (excludes id/created_at, which are DB-managed).
# INSERTABLE_COLUMNS = [
#     "filename",
#     "scrip",
#     "year",
#     "file_type",
#     "status",
#     "reason",
#     "destination_path",
#     "date",
#     "time"
# ]
INSERTABLE_COLUMNS = dataclass.fields(UploadResult)

# Validation
# Same pattern used in file_loader.py for Scrip — alphanumeric only.
SCRIP_PATTERN = re.compile(r"^[A-Za-z0-9]+$")

# 4-digit year.
YEAR_PATTERN = re.compile(r"^\d{4}$")

# Filetype is always "pdf" for now.
FILETYPE_PATTERN = re.compile(r"^pdf$")

# Full filename pattern, same as file_loader.py.
FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9]+_\d{4}_pdf\.pdf$")

TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")

DATE_PATTERN = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")

ALLOWED_STATUS_VALUES = {"SUCCESS", "FAILED"}

ALLOWED_TABLES = {"processed_files"}


# Max lengths — generous but bounded, to block abuse (huge strings, buffer-style abuse).
MAX_FILENAME_LENGTH = 255
MAX_REASON_LENGTH = 1000
MAX_PATH_LENGTH = 1024

# Characters that must never appear in a string going into the DB:
# - NUL byte: can truncate strings / confuse C-level string handling.
# - Newline / carriage return: log-injection risk if this value is ever
#   written to a log file (lets an attacker forge fake log lines).
# - SQL-meta characters are NOT filtered here, because we use parameterized
#   queries exclusively (see db.py) — that is the actual SQL-injection
#   defense. This list is about *secondary* injection vectors (path
#   traversal, log forging, control-character abuse), not SQL syntax.
FORBIDDEN_CHARS_PATTERN = re.compile(r"[\x00\r\n]")

# Path traversal indicators — destination_path should never contain these.
PATH_TRAVERSAL_PATTERN = re.compile(r"(\.\./|\.\.\\)")