"""All validation functions for the cleaning pipeline.

Convention: pure functions only — no I/O beyond what is strictly needed
to inspect the file on disk, no side-effects, no imports from sibling
modules other than skelton and exceptions.

Every public function either returns a validated value or raises a
domain exception. Callers never receive None — they either get a good
value or an exception they can handle.

Validation surface
------------------
1.  validate_input_filepath   — file selected by the user at pipeline entry
2.  validate_output_path      — cleaned / embedding paths written by the pipeline
3.  validate_filename         — standalone filename string (e.g. for DB inserts)
4.  validate_scrip            — stock symbol string
5.  validate_year             — year as int or coercible string
6.  validate_doc_type         — document type string
7.  validate_status           — DB status field ("SUCCESS" / "FAILED")
8.  validate_reason           — free-text reason / embedding_path stored in reason col
9.  validate_cleaned_json     — structure of the *_CLEANED.json written to disk
10. validate_embedding_json   — structure of the *_EMBEDDINGREADY.json written to disk
11. validate_db_insert_record — composite: validates every field before a DB insert
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codebase.cleaning.skelton import (
    ALLOWED_BASE,
    ALLOWED_EXTENSIONS,
    ALLOWED_DOC_TYPES,
    ALLOWED_STATUS_VALUES,
    FILENAME_RE,
    SCRIP_RE,
    MIN_YEAR,
    MAX_YEAR,
    MAX_PATH_LENGTH,
    MAX_REASON_LENGTH,
    MAX_SCRIP_LENGTH,
)
from codebase.cleaning.exceptions import FilePathError, CleaningPipelineError


# ---------------------------------------------------------------------------
# Internal shared helpers (not part of public API)
# ---------------------------------------------------------------------------

def _require_non_empty_string(value: object, label: str) -> str:
    """Return stripped string or raise CleaningPipelineError."""
    if value is None or not isinstance(value, str) or not value.strip():
        raise CleaningPipelineError(f"{label} must be a non-empty string, got: {value!r}")
    return value.strip()


def _check_path_length(path: object, label: str) -> None:
    if len(str(path)) > MAX_PATH_LENGTH:
        raise FilePathError(path, f"{label} exceeds maximum path length of {MAX_PATH_LENGTH} characters")


def _resolve_under_base(path: object, label: str) -> Path:
    """Resolve path and confirm it sits under ALLOWED_BASE."""
    _check_path_length(path, label)
    resolved = Path(str(path)).expanduser().resolve()
    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(path, f"{label} is outside the allowed base directory (possible path traversal)")
    return resolved


def _assert_file_exists(resolved: Path, original: object) -> None:
    if not resolved.exists():
        raise FilePathError(original, "File does not exist")
    if not resolved.is_file():
        raise FilePathError(original, "Path exists but is not a regular file")
    if resolved.stat().st_size == 0:
        raise FilePathError(original, "File is empty (0 bytes)")


def _parse_json_file(resolved: Path, original: object) -> Any:
    """Read and parse a JSON file; raise FilePathError on any failure."""
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except UnicodeDecodeError:
        raise FilePathError(original, "File is not valid UTF-8")
    except json.JSONDecodeError as exc:
        raise FilePathError(original, f"File is not valid JSON: {exc}")


def _decompose_filename(filename: str) -> tuple[str, int, str]:
    """Split a validated filename into (scrip, year, doc_type) components."""
    stem       = Path(filename).stem          # e.g. KALYANKJIL_2025_ANNUAL_REPORT
    parts      = stem.split("_", 2)           # max 3 parts: scrip, year, doc_type
    scrip      = parts[0]
    year       = int(parts[1])
    doc_type   = parts[2].upper()
    return scrip, year, doc_type


# ---------------------------------------------------------------------------
# 1. validate_input_filepath
#    Called first — the user has selected a file from disk.
# ---------------------------------------------------------------------------

def validate_input_filepath(path: object) -> Path:
    """Validate a user-supplied input file path end-to-end.

    Checks (in order):
    - Non-empty, within MAX_PATH_LENGTH
    - Resolves under ALLOWED_BASE (no path traversal)
    - Exists, is a regular file, non-empty
    - Correct directory depth: <base>/<scrip>/<year>/<filename>
    - Scrip folder matches SCRIP_RE
    - Year folder is numeric, within [MIN_YEAR, MAX_YEAR]
    - Filename matches FILENAME_RE (SCRIP_YEAR_DOCTYPE.json, case-insensitive)
    - Scrip in filename matches scrip folder name (case-insensitive)
    - Year in filename matches year folder
    - Extension is in ALLOWED_EXTENSIONS
    - File is valid UTF-8 JSON

    Returns the resolved Path.
    """
    if path is None or str(path).strip() == "":
        raise FilePathError(path, "File path is empty")

    resolved = _resolve_under_base(path, "Input file path")
    _assert_file_exists(resolved, path)

    relative_parts = resolved.relative_to(ALLOWED_BASE).parts
    if len(relative_parts) != 3:
        raise FilePathError(
            path,
            f"Expected structure <base>/<scrip>/<year>/<file> — got {len(relative_parts)} segment(s) below base",
        )

    folder_scrip, folder_year_str, filename = relative_parts

    # Scrip folder
    if not SCRIP_RE.match(folder_scrip):
        raise FilePathError(path, f"Scrip folder name is invalid: '{folder_scrip}'")

    # Year folder
    if not folder_year_str.isdigit():
        raise FilePathError(path, f"Year folder is not numeric: '{folder_year_str}'")
    folder_year = int(folder_year_str)
    if not (MIN_YEAR <= folder_year <= MAX_YEAR):
        raise FilePathError(
            path,
            f"Year folder {folder_year} is outside the plausible range ({MIN_YEAR}–{MAX_YEAR})",
        )

    # Filename pattern
    if not FILENAME_RE.match(filename):
        raise FilePathError(
            path,
            f"Filename '{filename}' does not match expected pattern "
            f"SCRIP_YEAR_DOCTYPE.json (e.g. KALYANKJIL_2025_ANNUAL_REPORT.json)",
        )

    # Consistency: scrip and year in filename must match folder names
    file_scrip, file_year, _ = _decompose_filename(filename)
    if file_scrip.upper() != folder_scrip.upper():
        raise FilePathError(
            path,
            f"Scrip in filename ('{file_scrip}') does not match scrip folder ('{folder_scrip}')",
        )
    if file_year != folder_year:
        raise FilePathError(
            path,
            f"Year in filename ({file_year}) does not match year folder ({folder_year})",
        )

    # Extension
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise FilePathError(
            path,
            f"Unsupported extension '{suffix}', expected one of {ALLOWED_EXTENSIONS}",
        )

    # Content
    _parse_json_file(resolved, path)

    return resolved


# ---------------------------------------------------------------------------
# 2. validate_output_path
#    Called before writing cleaned / embedding files produced by the pipeline.
# ---------------------------------------------------------------------------

def validate_output_path(path: object) -> Path:
    """Validate a pipeline-generated output file path.

    Checks:
    - Non-empty, within MAX_PATH_LENGTH
    - Resolves under ALLOWED_BASE (no traversal)
    - Extension is .json
    - Parent directory exists (pipeline must have created it already)

    Does NOT require the file to exist yet (it is about to be written).
    """
    if path is None or str(path).strip() == "":
        raise FilePathError(path, "Output path is empty")

    resolved = _resolve_under_base(path, "Output path")

    if resolved.suffix.lower() != ".json":
        raise FilePathError(path, f"Output file must be a .json file, got '{resolved.suffix}'")

    if not resolved.parent.exists():
        raise FilePathError(path, f"Output directory does not exist: '{resolved.parent}'")

    return resolved


# ---------------------------------------------------------------------------
# 3. validate_filename
#    Validates a bare filename string (no directory component).
# ---------------------------------------------------------------------------

def validate_filename(filename: object) -> str:
    """Return the validated filename string or raise CleaningPipelineError.

    Checks:
    - Non-empty string
    - No directory separators (bare filename only)
    - Matches FILENAME_RE
    """
    name = _require_non_empty_string(filename, "Filename")

    if "/" in name or "\\" in name:
        raise CleaningPipelineError(f"Filename must not contain directory separators: '{name}'")

    if not FILENAME_RE.match(name):
        raise CleaningPipelineError(
            f"Filename '{name}' does not match expected pattern "
            f"SCRIP_YEAR_DOCTYPE.json (e.g. KALYANKJIL_2025_ANNUAL_REPORT.json)"
        )

    return name


# ---------------------------------------------------------------------------
# 4. validate_scrip
# ---------------------------------------------------------------------------

def validate_scrip(scrip: object) -> str:
    """Return uppercased scrip string or raise CleaningPipelineError."""
    value = _require_non_empty_string(scrip, "Scrip")

    if len(value) > MAX_SCRIP_LENGTH:
        raise CleaningPipelineError(
            f"Scrip '{value}' exceeds maximum length of {MAX_SCRIP_LENGTH} characters"
        )

    if not SCRIP_RE.match(value):
        raise CleaningPipelineError(
            f"Scrip '{value}' is invalid — must be alphanumeric only (1–{MAX_SCRIP_LENGTH} chars)"
        )

    return value.upper()


# ---------------------------------------------------------------------------
# 5. validate_year
# ---------------------------------------------------------------------------

def validate_year(year: object) -> int:
    """Return year as int or raise CleaningPipelineError."""
    if year is None:
        raise CleaningPipelineError("Year must not be None")

    try:
        year_int = int(year)
    except (ValueError, TypeError):
        raise CleaningPipelineError(f"Year '{year}' is not a valid integer")

    if not (MIN_YEAR <= year_int <= MAX_YEAR):
        raise CleaningPipelineError(
            f"Year {year_int} is outside the plausible range ({MIN_YEAR}–{MAX_YEAR})"
        )

    return year_int


# ---------------------------------------------------------------------------
# 6. validate_doc_type
# ---------------------------------------------------------------------------

def validate_doc_type(doc_type: object) -> str:
    """Return uppercased doc_type or raise CleaningPipelineError."""
    value = _require_non_empty_string(doc_type, "Doc type")
    upper = value.upper()

    if upper not in ALLOWED_DOC_TYPES:
        raise CleaningPipelineError(
            f"Doc type '{value}' is not recognised. Allowed: {sorted(ALLOWED_DOC_TYPES)}"
        )

    return upper


# ---------------------------------------------------------------------------
# 7. validate_status
# ---------------------------------------------------------------------------

def validate_status(status: object) -> str:
    """Return uppercased status or raise CleaningPipelineError."""
    value = _require_non_empty_string(status, "Status")
    upper = value.upper()

    if upper not in ALLOWED_STATUS_VALUES:
        raise CleaningPipelineError(
            f"Status '{value}' is not valid. Allowed: {sorted(ALLOWED_STATUS_VALUES)}"
        )

    return upper


# ---------------------------------------------------------------------------
# 8. validate_reason
#    Used for both the free-text failure reason and the embedding_path that
#    is stored in the reason column on success.
# ---------------------------------------------------------------------------

def validate_reason(reason: object) -> str:
    """Return stripped reason string or raise CleaningPipelineError.

    Allows empty string (no reason on success rows where reason holds
    embedding_path instead — that path is validated separately via
    validate_output_path before being stored here).
    """
    if reason is None:
        raise CleaningPipelineError("Reason must not be None (use empty string if not applicable)")

    if not isinstance(reason, str):
        raise CleaningPipelineError(f"Reason must be a string, got {type(reason).__name__}")

    if len(reason) > MAX_REASON_LENGTH:
        raise CleaningPipelineError(
            f"Reason exceeds maximum length of {MAX_REASON_LENGTH} characters"
        )

    return reason.strip()


# ---------------------------------------------------------------------------
# 9. validate_cleaned_json
#    Called after the *_CLEANED.json is written before it is read by
#    EmbeddingPrepared or recorded in the DB.
# ---------------------------------------------------------------------------

_REQUIRED_CLEAN_PAGE_FIELDS: frozenset[str] = frozenset({
    "page_num",
    "original_text",
    "cleaned_text",
    "word_count",
    "is_short",
    "has_table",
})

def validate_cleaned_json(path: object) -> list[dict]:
    """Load and validate a *_CLEANED.json file.

    Checks:
    - Path resolves under ALLOWED_BASE, is a non-empty .json file
    - Content is a non-empty JSON array
    - Every element is a dict
    - Every element contains the required CleanResult fields
    - page_num is an integer
    - cleaned_text is a string

    Returns the parsed list of page dicts.
    """
    resolved = _resolve_under_base(path, "Cleaned JSON path")
    _assert_file_exists(resolved, path)

    data = _parse_json_file(resolved, path)

    if not isinstance(data, list):
        raise CleaningPipelineError(
            f"Cleaned JSON must be a list of page records, got {type(data).__name__}"
        )

    if len(data) == 0:
        raise CleaningPipelineError("Cleaned JSON contains no page records")

    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise CleaningPipelineError(
                f"Cleaned JSON record at index {idx} is not a dict (got {type(record).__name__})"
            )

        missing = _REQUIRED_CLEAN_PAGE_FIELDS - record.keys()
        if missing:
            raise CleaningPipelineError(
                f"Cleaned JSON record at index {idx} is missing required fields: {sorted(missing)}"
            )

        if not isinstance(record["page_num"], int):
            raise CleaningPipelineError(
                f"Cleaned JSON record at index {idx}: 'page_num' must be an int, "
                f"got {type(record['page_num']).__name__}"
            )

        if not isinstance(record["cleaned_text"], str):
            raise CleaningPipelineError(
                f"Cleaned JSON record at index {idx}: 'cleaned_text' must be a string, "
                f"got {type(record['cleaned_text']).__name__}"
            )

    return data


# ---------------------------------------------------------------------------
# 10. validate_embedding_json
#     Called after the *_EMBEDDINGREADY.json is written, before DB insert.
# ---------------------------------------------------------------------------

_REQUIRED_CHUNK_FIELDS: frozenset[str] = frozenset({"id", "text", "metadata"})

def validate_embedding_json(path: object) -> dict:
    """Load and validate a *_EMBEDDINGREADY.json file.

    Checks:
    - Path resolves under ALLOWED_BASE, is a non-empty .json file
    - Content is a dict with 'parents' and 'children' keys
    - Both values are lists
    - Every parent and child record contains 'id', 'text', 'metadata'
    - 'id' is a non-empty string
    - 'text' is a non-empty string
    - 'metadata' is a dict

    Returns the parsed bundle dict.
    """
    resolved = _resolve_under_base(path, "Embedding JSON path")
    _assert_file_exists(resolved, path)

    data = _parse_json_file(resolved, path)

    if not isinstance(data, dict):
        raise CleaningPipelineError(
            f"Embedding JSON must be a dict with 'parents' and 'children', got {type(data).__name__}"
        )

    for key in ("parents", "children"):
        if key not in data:
            raise CleaningPipelineError(f"Embedding JSON is missing required key: '{key}'")
        if not isinstance(data[key], list):
            raise CleaningPipelineError(
                f"Embedding JSON '{key}' must be a list, got {type(data[key]).__name__}"
            )

    for group in ("parents", "children"):
        for idx, chunk in enumerate(data[group]):
            if not isinstance(chunk, dict):
                raise CleaningPipelineError(
                    f"Embedding JSON '{group}[{idx}]' is not a dict"
                )

            missing = _REQUIRED_CHUNK_FIELDS - chunk.keys()
            if missing:
                raise CleaningPipelineError(
                    f"Embedding JSON '{group}[{idx}]' is missing required fields: {sorted(missing)}"
                )

            if not isinstance(chunk["id"], str) or not chunk["id"].strip():
                raise CleaningPipelineError(
                    f"Embedding JSON '{group}[{idx}]': 'id' must be a non-empty string"
                )

            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise CleaningPipelineError(
                    f"Embedding JSON '{group}[{idx}]': 'text' must be a non-empty string"
                )

            if not isinstance(chunk["metadata"], dict):
                raise CleaningPipelineError(
                    f"Embedding JSON '{group}[{idx}]': 'metadata' must be a dict"
                )

    return data


# ---------------------------------------------------------------------------
# 11. validate_db_insert_record
#     Composite — validates every field that will be written to the DB.
#     Called by db.insert_cleaning_record before the SQL executes.
# ---------------------------------------------------------------------------

def validate_db_insert_record(
    filename:       object,
    scrip:          object,
    year:           object,
    cleaned_path:   object,
    embedding_path: object,
    status:         object,
    reason:         object,
) -> tuple[str, str, int, str, str, str, str]:
    """Validate every field destined for the processed_files table.

    Returns a tuple of validated values in the same order as the parameters:
    (filename, scrip, year, cleaned_path, embedding_path, status, reason)

    Raises CleaningPipelineError or FilePathError on the first failure.
    """
    v_filename       = validate_filename(filename)
    v_scrip          = validate_scrip(scrip)
    v_year           = validate_year(year)
    v_cleaned_path   = str(validate_output_path(cleaned_path))
    v_embedding_path = str(validate_output_path(embedding_path))
    v_status         = validate_status(status)
    v_reason         = validate_reason(reason)

    # Cross-field: scrip and year in filename must match the record values
    file_scrip, file_year, _ = _decompose_filename(v_filename)
    if file_scrip.upper() != v_scrip.upper():
        raise CleaningPipelineError(
            f"Scrip in filename ('{file_scrip}') does not match scrip field ('{v_scrip}')"
        )
    if file_year != v_year:
        raise CleaningPipelineError(
            f"Year in filename ({file_year}) does not match year field ({v_year})"
        )

    return v_filename, v_scrip, v_year, v_cleaned_path, v_embedding_path, v_status, v_reason