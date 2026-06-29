"""Custom error types used across the fileloader module."""
from pathlib import Path
from typing import Any

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

class DatabaseInsertError(Exception):
    """Raised when a value about to be written to the DB fails upload."""

    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid value for '{field}': {reason} (got: {value!r})")