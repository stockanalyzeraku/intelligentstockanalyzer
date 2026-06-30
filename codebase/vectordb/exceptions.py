"""Domain exceptions for the vectordb module."""


class VectorDBError(Exception):
    """Base class for all vectordb errors."""


class UpsertError(VectorDBError):
    """Raised when writing records into a Chroma collection fails."""


class QueryError(VectorDBError):
    """Raised when querying a Chroma collection fails."""