"""
db.py — call-site for marking a PDF's OCR result in the processed_files table.

This file owns nothing about the database itself. The schema, the table,
and the actual write function (update_ocr_status) all live in
codebase/fileloader/db.py — fileloader is the single owner of the
processed_files database, exactly as before.

This file's only job is: given the PDF path the OCR pipeline just ran on,
work out which row that corresponds to, and call fileloader's
update_ocr_status with the right values. ocrprocessor.py calls these two
functions; nothing else in ocrprocessor needs to know how identity is
derived or how the database call is made.
"""
from __future__ import annotations

import os
import re

from codebase.fileloader import update_ocr_status, RecordNotFoundError

# Matches e.g. "KALYANKJIL_2023_ANNUAL_REPORT.pdf" → scrip / year / file_type.
# scrip is alphanumeric only (no underscore — matches fileloader's SCRIP_PATTERN),
# year is exactly 4 digits, file_type is everything remaining before the
# extension (may itself contain underscores, e.g. "ANNUAL_REPORT").
# Folder structure is not used — only the filename itself is parsed, since
# the uploads folder structure is not reliable yet.
_PDF_FILENAME_PATTERN = re.compile(
    r"^(?P<scrip>[A-Za-z0-9]+)_(?P<year>\d{4})_(?P<file_type>.+)\.pdf$"
)


def _parse_identity_from_filename(pdf_path: str) -> tuple[str, str, str]:
    """
    Derive (scrip, year, file_type) from the PDF's own filename.

    Args:
        pdf_path: Path to the source PDF (e.g. ".../KALYANKJIL_2023_ANNUAL_REPORT.pdf").

    Returns:
        (scrip, year, file_type) parsed from the filename.

    Raises:
        ValueError: if the filename does not match the expected pattern.
    """
    basename = os.path.basename(pdf_path)
    match = _PDF_FILENAME_PATTERN.match(basename)
    if not match:
        raise ValueError(
            f"PDF filename '{basename}' does not match the expected "
            f"pattern '<SCRIP>_<YEAR>_<FILE_TYPE>.pdf'"
        )
    return match.group("scrip"), match.group("year"), match.group("file_type")


def mark_ocr_success(pdf_path: str) -> int:
    """
    Mark the processed_files row for `pdf_path` as ocr_status='SUCCESS'.

    Args:
        pdf_path: Path to the PDF that was just OCR'd successfully.

    Returns:
        The id of the row that was updated.

    Raises:
        ValueError:          if pdf_path's filename can't be parsed.
        RecordNotFoundError: if no matching row exists.
    """
    scrip, year, file_type = _parse_identity_from_filename(pdf_path)
    return update_ocr_status(scrip, year, file_type, ocr_status="SUCCESS", ocr_reason=None)


def mark_ocr_failed(pdf_path: str, reason: str) -> int:
    """
    Mark the processed_files row for `pdf_path` as ocr_status='FAILED'.

    Args:
        pdf_path: Path to the PDF whose OCR pipeline raised an exception.
        reason:   The error message to store in ocr_reason.

    Returns:
        The id of the row that was updated.

    Raises:
        ValueError:          if pdf_path's filename can't be parsed.
        RecordNotFoundError: if no matching row exists.
    """
    scrip, year, file_type = _parse_identity_from_filename(pdf_path)
    return update_ocr_status(scrip, year, file_type, ocr_status="FAILED", ocr_reason=reason)