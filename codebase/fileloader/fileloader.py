from __future__ import annotations

import io
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from logger import StructuredLogger
from codebase.fileloader.skelton import UploadResult
from codebase.fileloader.validator import (
    _validate_filename,
    _validate_filesize,
    _validate_pdf_structure
)
from codebase.fileloader.exceptions import DuplicateFileError, FilenameValidationError
from config import CONFIG





def get_or_create_upload_dir(scrip: str, year: str, report_type: str, logger: StructuredLogger) -> Path:
    """
    Resolve (and create if missing) the directory a file should live in,
    based on scrip/year/report_type. Raises OSError on failure (e.g.
    permissions, disk full, invalid path) — caller decides how to report it.
    """
    target_dir = Path(CONFIG.UPLOADS_PATH) / scrip / year / report_type
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            f"Could not create upload directory '{target_dir}': {exc}",
            scrip=scrip,
            year=year,
            report_type=report_type,
            step="prepare_destination_dir",
            outcome="failed",
        )
        raise
    return target_dir


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def upload_file(file_bytes: bytes, filename: str, logger: StructuredLogger) -> UploadResult | str:
    """
    Validate, store, and report on a single PDF upload.
    Returns:
        UploadResult on success.
        A short, user-facing failure string on any rejected step (current
        convention for this function) — full detail goes to the logs.
    """
    logger.event(
        f"{filename} : Upload started",
        filename=filename, step="upload_file", stage="start",
    )

    # Step 1: filename -> scrip / year / file_type
    try:
        scrip, year, file_type = _validate_filename(filename)
    except FilenameValidationError as exc:
        logger.event(
            f"{filename} : Filename validation failed: {exc}",
            filename=filename, step="filename_validation", outcome="failed",
        )
        return "Put a valid filename"

    logger.event(
        f"{filename} : Filename validated successfully "
        f"(scrip={scrip}, year={year}, file_type={file_type})",
        filename=filename, step="filename_validation", outcome="passed",
        scrip=scrip, year=year, file_type=file_type,
    )

    # Step 2: file size
    try:
        _validate_filesize(file_bytes, filename)
    except ValueError as exc:
        logger.event(
            f"{filename} : File size validation failed: {exc}",
            filename=filename, step="filesize_validation", outcome="failed",
            size_bytes=len(file_bytes),
        )
        return "Put a valid filesize"

    logger.event(
        f"{filename} : File size validated successfully ({len(file_bytes)} bytes)",
        filename=filename, step="filesize_validation", outcome="passed",
        size_bytes=len(file_bytes),
    )

    # Step 3: PDF structure (header, parseable, not encrypted, has pages)
    try:
        _validate_pdf_structure(file_bytes, filename)
    except ValueError as exc:
        logger.event(
            f"{filename} : PDF structure validation failed: {exc}",
            filename=filename, step="pdf_structure_validation", outcome="failed",
        )
        return "Put a valid file structure"

    logger.event(
        f"{filename} : PDF structure validated successfully",
        filename=filename, step="pdf_structure_validation", outcome="passed",
    )

    # Step 4: resolve destination directory
    try:
        destination_dir = get_or_create_upload_dir(scrip, year, file_type, logger)
    except OSError as exc:
        logger.event(
            f"{filename} : Could not prepare destination directory: {exc}",
            filename=filename, step="prepare_destination_dir", outcome="failed",
        )
        return "Cant upload right now"

    destination_path = destination_dir / filename

    # Step 5: duplicate check — must actually stop the upload, not just log it
    if destination_path.exists():
        logger.event(
            f"{filename} : Rejected — {DuplicateFileError(destination_path)}",
            filename=filename, step="duplicate_check", outcome="failed",
            destination_path=str(destination_path),
        )
        return "File already exists"

    logger.event(
        f"{filename} : No existing file at destination, safe to write",
        filename=filename, step="duplicate_check", outcome="passed",
        destination_path=str(destination_path),
    )

    # Step 6: write to disk
    try:
        destination_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.error(
            f"{filename} : Failed to write file to disk at '{destination_path}': {exc}",
            filename=filename, step="write_to_disk", outcome="failed",
            destination_path=str(destination_path),
        )
        return "Cant upload right now"

    logger.info(
        f"{filename} : Uploaded successfully to '{destination_path}'",
        filename=filename, step="write_to_disk", outcome="passed",
        destination_path=str(destination_path),
    )

    return UploadResult(
        filename=filename,
        status="SUCCESS",
        scrip=scrip,
        year=year,
        destination_path=str(destination_path),
        file_type=file_type,
    )


async def delete_file(filepath: str | Path, logger: StructuredLogger) -> bool:
    """
    Delete a previously uploaded file from disk.
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(
            f"Delete requested for non-existent file: '{path}'",
            filepath=str(path), step="delete", outcome="not_found",
        )
        return False

    try:
        path.unlink()
        logger.info(
            f"File '{path}' deleted successfully.",
            filepath=str(path), step="delete", outcome="passed",
        )
        return True
    except OSError as exc:
        logger.error(
            f"Failed to delete file '{path}': {exc}",
            filepath=str(path), step="delete", outcome="failed",
        )
        raise