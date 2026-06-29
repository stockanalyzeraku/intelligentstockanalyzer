"""
Public entry point for the fileloader module: upload/delete a file, save
its record, and look records up. Everything exported here is meant to be
used by other parts of the app.
"""
from codebase.fileloader.fileloader import (upload_file, delete_file)
from codebase.fileloader.exceptions import DuplicateFileError,FilenameValidationError, DatabaseValidationError
from .db import (
    init_db,
    insert_upload_record,
    get_record_by_filename,
    get_records_by_scrip,
    get_records_by_status,
    get_all_records,
    get_records_by_date_range,
    DatabaseValidationError,
)


__all__ = [
    "upload_file",
    "delete_file",
    "DuplicateFileError",
    "FilenameValidationError",
    "init_db",
    "insert_upload_record",
    "get_record_by_filename",
    "get_records_by_scrip",
    "get_records_by_status",
    "get_all_records",
    "get_records_by_date_range",
    "DatabaseValidationError",
]