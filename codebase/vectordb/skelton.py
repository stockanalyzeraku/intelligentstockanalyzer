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