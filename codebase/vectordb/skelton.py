"""Dataclasses and static data for the vectordb module.

These shapes are the contract between db.py / store.py / retriever.py —
every piece that moves records in or out of Chroma agrees on one of these
instead of passing ad-hoc dicts around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Static, never-changing settings for the Chroma collections ──────────────
HNSW_SPACE: str = "cosine"
DEFAULT_UPSERT_BATCH_SIZE: int = 100
SCALAR_METADATA_TYPES: tuple[type, ...] = (str, int, float, bool)

# A payload is treated as a parent/child bundle when it carries both keys.
BUNDLE_KEYS: tuple[str, str] = ("parents", "children")

# ── Static validation thresholds / allow-lists for validator.py ─────────────
# Kept here (not hardcoded in validator.py) so every limit the module enforces
# is visible in one place, same as the rest of this file's static data.

COLLECTION_NAME_PATTERN: str = r"^[A-Za-z0-9_\-]{1,64}$"
MAX_COLLECTION_NAME_LENGTH: int = 64

MIN_RECORD_TEXT_LENGTH: int = 1
MAX_RECORD_TEXT_LENGTH: int = 20_000
MAX_RECORDS_PER_BATCH: int = 50_000
MAX_UPSERT_BATCH_SIZE: int = 500

MIN_QUERY_TEXT_LENGTH: int = 1
MAX_QUERY_TEXT_LENGTH: int = 2_000
MAX_QUERY_TEXTS: int = 5

MIN_TOP_K: int = 1
MAX_TOP_K: int = 100

MAX_CHUNK_IDS_PER_LOOKUP: int = 200

# Chroma 'where' filter safety — this is Chroma-specific domain knowledge,
# so it's owned here rather than in a generic, project-wide validator.
ALLOWED_WHERE_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "company",
        "financial_year",
        "report_year",
        "year",
        "doc_type",
        "source",
        "file_name",
        "document_name",
        "page_number",
        "page",
        "page_no",
        "parent_id",
    }
)
ALLOWED_WHERE_OPERATORS: frozenset[str] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$and", "$or"}
)
MAX_WHERE_FILTER_DEPTH: int = 3
MAX_WHERE_FILTER_LIST_SIZE: int = 50

# File-path inputs accepted from outside the module.
ALLOWED_EMBEDDING_JSON_SUFFIX: str = ".json"


@dataclass(frozen=True)
class ChromaRecord:
    """One record as stored in / fetched from a Chroma collection."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildMatch:
    """The best-scoring child chunk found for a given parent."""

    child_id: str
    child_text: str
    child_metadata: dict[str, Any]
    distance: float


@dataclass(frozen=True)
class RetrievedItem:
    """A parent section merged with its best-matching child chunk."""

    id: str
    text: str
    metadata: dict[str, Any]
    parent_id: str
    parent_text: str
    parent_metadata: dict[str, Any]
    child_id: str
    child_text: str
    child_metadata: dict[str, Any]
    distance: float

    def as_dict(self) -> dict[str, Any]:
        """Render as the plain dict shape existing callers already expect."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "parent_metadata": self.parent_metadata,
            "child_id": self.child_id,
            "child_text": self.child_text,
            "child_metadata": self.child_metadata,
            "distance": self.distance,
        }