from __future__ import annotations

import io
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


from codebase.fileloader.skelton import UploadResult
from codebase.fileloader.validator import _validate_filename, _validate_filesize
from codebase.fileloader.exceptions import DuplicateFileError
from config import CONFIG

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF-"

# Internal validation helpers


def _validate_pdf_structure(file_bytes: bytes, filename: str) -> None:
    """
    Confirm the file is a genuine, parseable, non-encrypted PDF.
    Raises ValueError on any failure.
    """
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise ValueError(f"File '{filename}' is not a valid PDF (missing PDF header).")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as exc:
        raise ValueError(f"File '{filename}' could not be parsed as a PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any other parse failure means bogus file
        raise ValueError(f"File '{filename}' is not a valid/readable PDF: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError(f"File '{filename}' is encrypted/password-protected and is rejected.")

    if len(reader.pages) == 0:
        raise ValueError(f"File '{filename}' contains no pages and is rejected.")



def get_or_create_upload_dir(scrip: str, year: str, report_type: str) -> Path:

    target_dir = Path(CONFIG.UPLOADS_PATH) / scrip / year / report_type
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

async def upload_file(file_bytes: bytes, filename: str) -> UploadResult:
    scrip, year, file_type = _validate_filename(filename)
    try:
        _validate_filesize(file_bytes, filename)
        _validate_pdf_structure(file_bytes, filename)
    except ValueError as exc:
        return UploadResult(
            filename=filename,
            status="FAILED",
            reason=str(exc),
            scrip=scrip,
            year=year,
        )

    # Step 3: prepare destination folder
    destination_dir = get_or_create_upload_dir(scrip, year, file_type)
    destination_path = destination_dir / filename
    if destination_path.exists():
        return UploadResult(
            filename=filename,
            status="FAILED",
            reason=str(exc),
            scrip=scrip,
            year=year
            )


    try:
        destination_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.error("Failed to write file '%s' to disk: %s", filename, exc)
        return UploadResult(
            filename=filename,
            status="FAILED",
            reason=f"Failed to save file: {exc}",
            scrip=scrip,
            year=year,
            file_type = file_type
        )

    logger.info("File '%s' uploaded successfully to '%s'.", filename, destination_path)
    return UploadResult(
        filename=filename,
        status="SUCCESS",
        scrip=scrip,
        year=year,
        destination_path=str(destination_path),
        file_type = file_type
    )


async def delete_file(filepath: str | Path) -> bool:
    """
    Delete a previously uploaded file from disk.
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning("Delete requested for non-existent file: '%s'", path)
        return False

    try:
        path.unlink()
        logger.info("File '%s' deleted successfully.", path)
        return True
    except OSError as exc:
        logger.error("Failed to delete file '%s': %s", path, exc)
        raise