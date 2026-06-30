"""Domain exceptions for the vectordb module."""


class VectorDBError(Exception):
    """Base class for all vectordb errors."""


class UpsertError(VectorDBError):
    """Raised when writing records into a Chroma collection fails."""


class QueryError(VectorDBError):
    """Raised when querying a Chroma collection fails."""


# ── Validation errors ────────────────────────────────────────────────────────
# Raised by validator.py. Split by what failed so callers (and logs, once
# error handling is wired up) can tell a malformed collection name apart
# from a duplicate record id apart from an unsafe filter, instead of
# catching one generic ValidationError for everything.

class VectorDBValidationError(VectorDBError):
    """Base class for all input-validation failures in this module."""


class InvalidCollectionNameError(VectorDBValidationError):
    """Raised when a collection name fails the allowed name format/length."""


class InvalidRecordError(VectorDBValidationError):
    """Raised when a single record (id/text/metadata) is malformed."""


class DuplicateRecordIdError(VectorDBValidationError):
    """Raised when a batch of records contains the same id more than once."""


class InvalidMetadataError(VectorDBValidationError):
    """Raised when record or filter metadata contains disallowed shapes/types."""


class InvalidQueryTextError(VectorDBValidationError):
    """Raised when query text input is empty, too long, or the wrong type."""


class InvalidTopKError(VectorDBValidationError):
    """Raised when a requested result count is out of the allowed range."""


class UnsafeFilterError(VectorDBValidationError):
    """Raised when a Chroma 'where' filter uses a disallowed key/operator/shape."""


class InvalidPathError(VectorDBValidationError):
    """Raised when a filesystem path input is unsafe, missing, or wrong type."""


class InvalidPayloadShapeError(VectorDBValidationError):
    """Raised when an embedding JSON payload isn't a bundle or a flat record list."""


class InvalidIdError(VectorDBValidationError):
    """Raised when a chunk/record id used for lookup is malformed."""


class EmbeddingDimensionError(VectorDBValidationError):
    """Raised when an embedding vector's dimensionality doesn't match expectations."""