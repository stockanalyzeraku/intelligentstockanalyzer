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
    TIME_PATTERN
)
from codebase.fileloader.skelton import(
    FILENAME_PATTERN
)
from datetime import datetime
from config import CONFIG

# Field validation — defense in depth, run again right before insert.
def _check_forbidden_chars(field: str, value: str) -> None:
    if FORBIDDEN_CHARS_PATTERN.search(value):
        raise DatabaseValidationError(
            field, value, "contains forbidden control characters (NUL/CR/LF)."
        )
    
#validate filename
def _validate_filename(filename:str) -> tuple[str, str, str]:
    if not filename or not isinstance(filename, str):
        raise FilenameValidationError(filename, "must be a non-empty string.")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise FilenameValidationError( filename, "exceeds max length.")
    _check_forbidden_chars("filename", filename)
    match = FILENAME_PATTERN.match(filename)
    if not FILENAME_PATTERN.match(filename):
        raise FilenameValidationError(
            filename, "does not match required Scrip_Year_pdf.pdf pattern."
        )
    return match.group("scrip"), match.group("year"), match.group("filetype")
 
#validate scrip
def _validate_scrip(scrip:str | None) -> None:
    if scrip is not None:
        if not isinstance(scrip, str) or not SCRIP_PATTERN.match(scrip):
            raise DatabaseValidationError("scrip", scrip, "must be alphanumeric.")
        _check_forbidden_chars("scrip", scrip)
    else:
        raise DatabaseValidationError("scrip", scrip, "scrip is empty")

def _validate_limit(limit:int = 1000)-> None:
    if not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if not (1 <= limit <= 1000):
        raise ValueError("limit must be between 1 and 1000.")

#parse date
def _validate_parse_date(date: str, field: str) -> datetime:
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
    if len(file_bytes) == 0:
        raise ValueError(f"File '{filename}' is empty.")
    if len(file_bytes) > CONFIG.MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File '{filename}' exceeds the maximum allowed size of "
            f"{CONFIG.MAX_FILE_SIZE_MB}MB."
        )

def _validate_year(year: str) -> None:
    if year is not None:
        if not isinstance(year, str) or not YEAR_PATTERN.match(year):
            raise DatabaseValidationError("year", year, "must be a 4-digit string.")

def _validate_filetype(filetype: str) -> None:
    if filetype is not None:
        if not isinstance(filetype, str) or not FILETYPE_PATTERN.match(filetype):
            raise DatabaseValidationError("filetype", filetype, "must be 'pdf'.")
    
def _validate_status(status) -> None:
    if status is not None:
        if not isinstance(status, str) or not ALLOWED_STATUS_VALUES.MATCH(status):
            raise DatabaseValidationError("status", status, "Not a valid status'.")

def _validate_reason(reason: str) -> None:
    if reason is not None:
        if not isinstance(reason, str):
            raise DatabaseValidationError("reason", reason, "must be a string.")
        if len(reason) > MAX_REASON_LENGTH:
            raise DatabaseValidationError("reason", reason, "exceeds max length.")
        _check_forbidden_chars("reason", reason)

def _validate_destination_path(destination_path: str) -> None:
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
    if not time or not isinstance(time, str) or not TIME_PATTERN.match(time):
            raise DatabaseValidationError(
                "upload_time", time, "must be a non-empty string"
            )
    _check_forbidden_chars("upload_time", time)
        
    
