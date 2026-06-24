"""Custom exceptions for the application."""

from pathlib import Path
from typing import Any

class Error(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class ValidationError(Error):
    """Raised when input validation fails."""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class ProcessingError(Error):
    """Raised when document processing fails."""
    def __init__(self, message: str):
        super().__init__(message, "PROCESSING_ERROR")

class CacheError(Error):
    """Raised when cache operations fail."""
    def __init__(self, message: str):
        super().__init__(message, "CACHE_ERROR")

class ExternalAPIError(Error):
    """Raised when external API calls fail."""
    def __init__(self, message: str):
        super().__init__(message, "EXTERNAL_API_ERROR")

class TimeoutError(Error):
    """Raised when an operation times out."""
    def __init__(self, message: str):
        super().__init__(message, "TIMEOUT")

#Load New FIle
class FilenameValidationError(Exception):
    """Raised when the uploaded filename does not match the required pattern."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Invalid filename '{filename}': {reason}")

class DuplicateFileError(Exception):
    """Raised when a file with the same name already exists at the destination."""

    def __init__(self, destination: Path):
        self.destination = destination
        super().__init__(f"File already exists at '{destination}'. Overwrite not allowed.")


class DatabaseValidationError(Exception):
    """Raised when a value about to be written to the DB fails validation."""

    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid value for '{field}': {reason} (got: {value!r})")

