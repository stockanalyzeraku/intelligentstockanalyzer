"""All input validation for the vectordb module.

Two categories of input get validated here, matching the project
convention (every input going into the database, and every input
coming from outside the module, must be validated):

  1. Data destined for Chroma — record ids/text/metadata, batches of
     records, embedding vectors. These run right before anything is
     written, so nothing malformed ever reaches db.py/store.py.

  2. Input arriving from outside this module — collection names,
     query texts, top_k, where-filters, chunk ids, and the JSON/Chroma
     filesystem paths passed into the ChromaStore facade.

This file does not call into db.py, store.py, retriever.py, or
chromastore.py, and nothing in those files calls into this file yet —
wiring validation into the actual code paths is the next step. Each
function below simply takes a raw input and either returns a cleaned/
validated value or raises one of the VectorDBValidationError subclasses
from exceptions.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from codebase.vectordb.exceptions import (
    DuplicateRecordIdError,
    EmbeddingDimensionError,
    InvalidCollectionNameError,
    InvalidIdError,
    InvalidMetadataError,
    InvalidPathError,
    InvalidPayloadShapeError,
    InvalidQueryTextError,
    InvalidRecordError,
    InvalidTopKError,
    UnsafeFilterError,
)
from codebase.vectordb.skelton import (
    ALLOWED_EMBEDDING_JSON_SUFFIX,
    ALLOWED_WHERE_FILTER_KEYS,
    ALLOWED_WHERE_OPERATORS,
    BUNDLE_KEYS,
    COLLECTION_NAME_PATTERN,
    MAX_CHUNK_IDS_PER_LOOKUP,
    MAX_QUERY_TEXT_LENGTH,
    MAX_QUERY_TEXTS,
    MAX_RECORD_TEXT_LENGTH,
    MAX_RECORDS_PER_BATCH,
    MAX_TOP_K,
    MAX_UPSERT_BATCH_SIZE,
    MAX_WHERE_FILTER_DEPTH,
    MAX_WHERE_FILTER_LIST_SIZE,
    MIN_QUERY_TEXT_LENGTH,
    MIN_RECORD_TEXT_LENGTH,
    MIN_TOP_K,
    SCALAR_METADATA_TYPES,
)

_COLLECTION_NAME_RE = re.compile(COLLECTION_NAME_PATTERN)


# ============================================================================
# Collection name
# ============================================================================

def validate_collection_name(collection_name: Any) -> str:
    """Validate a Chroma collection name (used by every store/query/lookup call)."""
    if not isinstance(collection_name, str):
        raise InvalidCollectionNameError("Collection name must be a string.")
    if not _COLLECTION_NAME_RE.match(collection_name):
        raise InvalidCollectionNameError(
            f"Invalid collection name '{collection_name}'. "
            "Must be 1-64 characters of letters, digits, '_' or '-'."
        )
    return collection_name


# ============================================================================
# Records going into the database (upsert path)
# ============================================================================

def validate_record_id(record_id: Any) -> str:
    """Validate a single record's id before it can be written to Chroma."""
    if not isinstance(record_id, str) or not record_id.strip():
        raise InvalidRecordError("Record id must be a non-empty string.")
    return record_id


def validate_record_text(text: Any) -> str:
    """Validate a single record's document text before it can be written to Chroma."""
    if not isinstance(text, str):
        raise InvalidRecordError("Record text must be a string.")
    stripped = text.strip()
    if len(stripped) < MIN_RECORD_TEXT_LENGTH:
        raise InvalidRecordError("Record text must not be empty.")
    if len(text) > MAX_RECORD_TEXT_LENGTH:
        raise InvalidRecordError(
            f"Record text too long ({len(text)} chars, max {MAX_RECORD_TEXT_LENGTH})."
        )
    return text


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """Validate metadata attached to a record before it can be written to Chroma.

    Chroma only stores scalar values, so every key must map to a
    str/int/float/bool. Catching non-scalars here (instead of silently
    dropping them, which is what sanitize_metadata in schemas.py does
    today) means bad metadata fails loudly at the boundary rather than
    disappearing without the caller noticing.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise InvalidMetadataError("Metadata must be a dictionary.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise InvalidMetadataError(f"Metadata keys must be non-empty strings, got {key!r}.")
        if value is not None and not isinstance(value, SCALAR_METADATA_TYPES):
            raise InvalidMetadataError(
                f"Metadata value for '{key}' must be str/int/float/bool, got {type(value).__name__}."
            )
    return metadata


def validate_record(raw_record: Any) -> dict[str, Any]:
    """Validate one raw {id, text, metadata} record dict before upsert."""
    if not isinstance(raw_record, dict):
        raise InvalidRecordError(f"Record must be an object, got {type(raw_record).__name__}.")
    record_id = validate_record_id(raw_record.get("id"))
    text = validate_record_text(raw_record.get("text"))
    metadata = validate_metadata(raw_record.get("metadata"))
    return {"id": record_id, "text": text, "metadata": metadata}


def validate_records(raw_records: Any) -> list[dict[str, Any]]:
    """Validate a full batch of raw records before upsert.

    Checks shape/list-ness, validates every record individually, and
    rejects duplicate ids within the same batch — a duplicate id here
    means one write would silently overwrite another in the same call.
    """
    if not isinstance(raw_records, list):
        raise InvalidRecordError(f"Records must be a list, got {type(raw_records).__name__}.")
    if len(raw_records) > MAX_RECORDS_PER_BATCH:
        raise InvalidRecordError(
            f"Too many records in one batch ({len(raw_records)}, max {MAX_RECORDS_PER_BATCH})."
        )

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        try:
            record = validate_record(raw_record)
        except InvalidRecordError as exc:
            raise InvalidRecordError(f"Record at index {index} is invalid: {exc}") from exc
        if record["id"] in seen_ids:
            raise DuplicateRecordIdError(f"Duplicate record id in batch: '{record['id']}'.")
        seen_ids.add(record["id"])
        validated.append(record)
    return validated


def validate_batch_size(batch_size: Any) -> int:
    """Validate the upsert batch size used to chunk writes to Chroma."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise InvalidRecordError("batch_size must be an integer.")
    if batch_size < 1 or batch_size > MAX_UPSERT_BATCH_SIZE:
        raise InvalidRecordError(f"batch_size must be between 1 and {MAX_UPSERT_BATCH_SIZE}.")
    return batch_size


# ============================================================================
# Embedding JSON payload (outside input: file content for store_in_chromadb)
# ============================================================================

def validate_embedding_payload(payload: Any) -> Any:
    """Validate the top-level shape of a loaded embedding JSON payload.

    Accepts either a flat list of records, or a {"parents": [...],
    "children": [...]} bundle. Per-record validation is delegated to
    validate_records so the rules aren't duplicated.
    """
    if isinstance(payload, dict) and all(key in payload for key in BUNDLE_KEYS):
        parents, children = payload.get("parents"), payload.get("children")
        validate_records(parents)
        validate_records(children)
        return payload
    if isinstance(payload, list):
        validate_records(payload)
        return payload
    raise InvalidPayloadShapeError(
        "Embedding payload must be a list of records or a parent/child bundle "
        f"with keys {BUNDLE_KEYS}."
    )


# ============================================================================
# Filesystem paths (outside input)
# ============================================================================

def validate_json_path(path: Any, *, base_dir: str | os.PathLike[str], must_exist: bool = True) -> str:
    """Validate a JSON file path passed in from outside, scoped to base_dir."""
    if not isinstance(path, (str, os.PathLike)):
        raise InvalidPathError("JSON path must be a path string.")
    resolved = Path(path).expanduser().resolve()
    base = Path(base_dir).expanduser().resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise InvalidPathError(f"Path is outside the allowed directory: '{resolved}'.") from exc
    if resolved.suffix.lower() != ALLOWED_EMBEDDING_JSON_SUFFIX:
        raise InvalidPathError(f"Path must end in '{ALLOWED_EMBEDDING_JSON_SUFFIX}': '{resolved}'.")
    if must_exist and not resolved.is_file():
        raise InvalidPathError(f"File not found: '{resolved}'.")
    return str(resolved)


def validate_chroma_path(path: Any) -> str:
    """Validate an on-disk path used as a Chroma persistence directory."""
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise InvalidPathError("Chroma path must be a non-empty path string.")
    return str(Path(path).expanduser())


# ============================================================================
# Query input (outside input)
# ============================================================================

def validate_query_texts(query_texts: Any) -> list[str]:
    """Validate the list of query strings passed in for a similarity search."""
    if not isinstance(query_texts, list) or not query_texts:
        raise InvalidQueryTextError("query_texts must be a non-empty list.")
    if len(query_texts) > MAX_QUERY_TEXTS:
        raise InvalidQueryTextError(f"Too many query texts (max {MAX_QUERY_TEXTS}).")

    validated: list[str] = []
    for index, text in enumerate(query_texts):
        if not isinstance(text, str):
            raise InvalidQueryTextError(f"query_texts[{index}] must be a string.")
        stripped = text.strip()
        if len(stripped) < MIN_QUERY_TEXT_LENGTH:
            raise InvalidQueryTextError(f"query_texts[{index}] must not be empty.")
        if len(text) > MAX_QUERY_TEXT_LENGTH:
            raise InvalidQueryTextError(
                f"query_texts[{index}] too long ({len(text)} chars, max {MAX_QUERY_TEXT_LENGTH})."
            )
        validated.append(text)
    return validated


def validate_top_k(top_k: Any, *, default: int, max_value: int | None = None) -> int:
    """Validate the requested number of results for a query."""
    value = default if top_k is None else top_k
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidTopKError("top_k must be an integer.")
    hard_max = max_value if max_value is not None else MAX_TOP_K
    if value < MIN_TOP_K or value > hard_max:
        raise InvalidTopKError(f"top_k must be between {MIN_TOP_K} and {hard_max}.")
    return value


def validate_where_filter(where: Any) -> dict[str, Any] | None:
    """Validate a Chroma metadata filter against allowed keys/operators/shape."""
    if where is None:
        return None
    if not isinstance(where, dict):
        raise UnsafeFilterError("where filter must be a dictionary.")

    def walk(node: Any, depth: int) -> None:
        if depth > MAX_WHERE_FILTER_DEPTH:
            raise UnsafeFilterError("where filter is nested too deeply.")
        if isinstance(node, dict):
            for key, value in node.items():
                if key.startswith("$"):
                    if key not in ALLOWED_WHERE_OPERATORS:
                        raise UnsafeFilterError(f"Unsupported where operator: '{key}'.")
                elif key not in ALLOWED_WHERE_FILTER_KEYS:
                    raise UnsafeFilterError(f"Unsupported where filter key: '{key}'.")
                walk(value, depth + 1)
        elif isinstance(node, list):
            if len(node) > MAX_WHERE_FILTER_LIST_SIZE:
                raise UnsafeFilterError("where filter list is too large.")
            for item in node:
                walk(item, depth + 1)
        elif node is not None and not isinstance(node, SCALAR_METADATA_TYPES):
            raise UnsafeFilterError("where filter values must be scalar, list, or dict.")

    walk(where, 0)
    return where


# ============================================================================
# Id lookups (outside input)
# ============================================================================

def validate_chunk_id(chunk_id: Any) -> str:
    """Validate a single id used for a get_by_id lookup."""
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise InvalidIdError("chunk_id must be a non-empty string.")
    return chunk_id


def validate_chunk_ids(chunk_ids: Any) -> list[str]:
    """Validate a list of ids used for a get_many_by_ids lookup."""
    if not isinstance(chunk_ids, list):
        raise InvalidIdError("chunk_ids must be a list.")
    if len(chunk_ids) > MAX_CHUNK_IDS_PER_LOOKUP:
        raise InvalidIdError(f"Too many chunk_ids in one lookup (max {MAX_CHUNK_IDS_PER_LOOKUP}).")
    return [validate_chunk_id(chunk_id) for chunk_id in chunk_ids]


# ============================================================================
# Embedding vectors (going into the database, computed just before upsert/query)
# ============================================================================

def validate_embedding_vector(vector: Any, *, expected_dim: int) -> list[float]:
    """Validate a single embedding vector's shape before it reaches Chroma."""
    if not isinstance(vector, (list, tuple)):
        raise EmbeddingDimensionError("Embedding vector must be a list of floats.")
    if len(vector) != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedding vector has dimension {len(vector)}, expected {expected_dim}."
        )
    if not all(isinstance(component, (int, float)) for component in vector):
        raise EmbeddingDimensionError("Embedding vector must contain only numeric values.")
    return list(vector)


def validate_embedding_vectors(vectors: Any, *, expected_dim: int) -> list[list[float]]:
    """Validate a batch of embedding vectors (e.g. one per query text)."""
    if not isinstance(vectors, list) or not vectors:
        raise EmbeddingDimensionError("Embedding vectors must be a non-empty list.")
    return [validate_embedding_vector(vector, expected_dim=expected_dim) for vector in vectors]
