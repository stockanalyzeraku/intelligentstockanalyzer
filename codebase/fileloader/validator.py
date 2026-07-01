"""
All input validation for the fileloader module lives here: filenames,
file size, PDF structure, and every field that goes into the database.
"""
from codebase.fileloader.exceptions import  (
    DatabaseValidationError,
    FilenameValidationError
)
from codebase.fileloader.schemas import (
    FORBIDDEN_CHARS_PATTERN,
    DATE_PATTERN,
    SCRIP_PATTERN,
    MAX_FILENAME_LENGTH,
    YEAR_PATTERN,
    FILETYPE_PATTERN,
    ALLOWED_STATUS_VALUES,
    MAX_REASON_LENGTH,
    MAX_PATH_LENGTH,
    PATH_TRAVERSAL_PATTERN,
    TIME_PATTERN,
    PDF_MAGIC_BYTES
)
from codebase.fileloader.skelton import(
    FILENAME_PATTERN
)
from datetime import datetime
from config import CONFIG
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import io

# Field validation — defense in depth, run again right before insert.
def _check_forbidden_chars(field: str, value: str) -> None:
    """Reject a value if it contains a NUL byte, carriage return, or newline."""
    if FORBIDDEN_CHARS_PATTERN.search(value):
        raise DatabaseValidationError(
            field, value, "contains forbidden control characters (NUL/CR/LF)."
        )

#validate filename
def _validate_filename(filename:str) -> tuple[str, str, str]:
    """
    Check that a filename matches the required Scrip_Year_pdf.pdf pattern.
    Returns (scrip, year, filetype) pulled out of the name if it's valid.
    """
    if not filename or not isinstance(filename, str):
        raise FilenameValidationError(filename, "must be a non-empty string.")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise FilenameValidationError( filename, "exceeds max length.")
    _check_forbidden_chars("filename", filename)
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise FilenameValidationError(
            filename, "does not match required Scrip_Year_pdf.pdf pattern."
        )
    return match.group("scrip"), match.group("year"), match.group("filetype")

#validate scrip
def _validate_scrip(scrip:str) -> None:
    """Check that a scrip (stock symbol) is a non-empty alphanumeric string."""
    if scrip is not None:
        if not isinstance(scrip, str) or not SCRIP_PATTERN.match(scrip):
            raise DatabaseValidationError("scrip", scrip, "must be alphanumeric.")
        _check_forbidden_chars("scrip", scrip)
    else:
        raise DatabaseValidationError("scrip", scrip, "scrip is empty")

def _validate_limit(limit:int = 1000)-> None:
    """Check that a query's row limit is a whole number between 1 and 1000."""
    if not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if not (1 <= limit <= 1000):
        raise ValueError("limit must be between 1 and 1000.")

#parse date
def _validate_parse_date(date: str, field: str) -> datetime:
    """Check that a date string is valid and parse it into a datetime."""
    if not date or not isinstance(date, str):
        raise DatabaseValidationError("date", date, "Empty date given or is not in string format")
    _check_forbidden_chars("date",date)
    for fmt in DATE_PATTERN:
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            continue
    raise DatabaseValidationError(field, date, "invalid date format.")


def _validate_filesize(file_bytes: bytes, filename: str) -> None:
    """Check that a file is not empty and not larger than the configured limit."""
    if len(file_bytes) == 0:
        raise ValueError(f"File '{filename}' is empty.")
    if len(file_bytes) > CONFIG.MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File '{filename}' exceeds the maximum allowed size of "
            f"{CONFIG.MAX_FILE_SIZE_MB}MB."
        )

def _validate_year(year: str) -> None:
    """Check that a year is a 4-digit string, if one was given."""
    if year is not None:
        if not isinstance(year, str) or not YEAR_PATTERN.match(year):
            raise DatabaseValidationError("year", year, "must be a 4-digit string.")

def _validate_filetype(filetype: str) -> None:
    """Check that a file type is exactly 'pdf', if one was given."""
    if filetype is not None:
        if not isinstance(filetype, str) or not FILETYPE_PATTERN.match(filetype):
            raise DatabaseValidationError("filetype", filetype, "must be 'pdf'.")

def _validate_status(status) -> None:
    """Check that a status is either 'SUCCESS' or 'FAILED', if one was given."""
    if status is not None:
        if not isinstance(status, str) or status not in ALLOWED_STATUS_VALUES:
            raise DatabaseValidationError("status", status, "Not a valid status'.")

def _validate_reason(reason: str) -> None:
    """Check that a reason string is short enough and contains no bad characters."""
    if reason is not None:
        if not isinstance(reason, str):
            raise DatabaseValidationError("reason", reason, "must be a string.")
        if len(reason) > MAX_REASON_LENGTH:
            raise DatabaseValidationError("reason", reason, "exceeds max length.")
        _check_forbidden_chars("reason", reason)

def _validate_destination_path(destination_path: str) -> None:
    """Check that a saved file's path is short, clean, and has no path-traversal tricks."""
    if destination_path is not None:
        if not isinstance(destination_path, str):
            raise DatabaseValidationError(
                "destination_path", destination_path, "must be a string."
            )
        if len(destination_path) > MAX_PATH_LENGTH:
            raise DatabaseValidationError("destination_path", destination_path, "exceeds max length.")
        _check_forbidden_chars("destination_path", destination_path)
        if PATH_TRAVERSAL_PATTERN.search(destination_path):
            raise DatabaseValidationError(
                "destination_path", destination_path, "contains path traversal sequence."
            )

def _validate_time(time: str) -> None:
    """Check that a time string is a real time of day, in HH:MM:SS format (00:00:00 to 23:59:59)."""
    if not time or not isinstance(time, str) or not TIME_PATTERN.match(time):
        raise DatabaseValidationError(
            "upload_time", time, "must be a non-empty string in HH:MM:SS format"
        )
    _check_forbidden_chars("upload_time", time)
    try:
        datetime.strptime(time, "%H:%M:%S")
    except ValueError:
        raise DatabaseValidationError(
            "upload_time", time, "must be a valid time between 00:00:00 and 23:59:59."
        )

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
    
ALLOWED_OCR_STATUS_VALUES = {"PENDING", "SUCCESS", "FAILED"}

def _validate_ocr_status(ocr_status: str) -> None:
    """Check that an OCR status is PENDING, SUCCESS, or FAILED."""
    if not isinstance(ocr_status, str) or ocr_status not in ALLOWED_OCR_STATUS_VALUES:
        raise DatabaseValidationError("ocr_status", ocr_status, "must be PENDING, SUCCESS or FAILED.")

def _validate_ocr_reason(ocr_reason: str | None) -> None:
    """Check that an OCR reason string is short and clean, if one is given."""
    if ocr_reason is not None:
        _validate_reason(ocr_reason)   # reuses the existing reason validator