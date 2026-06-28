from __future__ import annotations

import io
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from logger import StructuredLogger
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

def upload_file(file_bytes: bytes, filename: str, logger: StructuredLogger) -> UploadResult | str:
    scrip, year, file_type = _validate_filename(filename)
    is_valid_filename = all(v is not None for v in (scrip, year, file_type))
    if is_valid_filename:
        logger.event(f"{filename} : Validated file name, Successfull. Scrip : {scrip}, Year: {year}, File Type: {file_type}")
    else:
        logger.event(f"{filename} : Validated file Name, Not Succesfull. Scrip : {scrip}, Year: {year}, File Type: {file_type}")
        return "Put a valid filename"

    try:
        _validate_filesize(file_bytes, filename)
        logger.event(f"{filename} : File size is valid")
    except ValueError as exc:
        logger.event(f"{filename} : File size inappropriate : {exc}")
        return "Put a valid filesize"

    try: 
        _validate_pdf_structure(file_bytes, filename)
        logger.event(f"{filename} : File structure is valid")
    except ValueError as exc:
        logger.event(f"{filename} : File structure invalid : {exc}")
        return "Put a valid file structure"

    # Step 3: prepare destination folder
    destination_dir = get_or_create_upload_dir(scrip, year, file_type)
    destination_path = destination_dir / filename
    if destination_path.exists():
        logger.event(f"{filename} : destinantion folder created succesffully")
    else:
        logger.event(f"{filename} : destination folder creation failed")
        return "Cant upload right now"
    
    try:
        destination_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.event("Failed to write file '%s' to disk: %s", filename, exc)
        return "Cant upload right now"

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