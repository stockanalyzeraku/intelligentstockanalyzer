"""Shape detection and normalisation for embedding-ready payloads.

Deep field validation (ids present, no duplicates, text non-empty, ...)
stays in the shared InputValidator — that's a cross-cutting input-safety
concern, not something specific to Chroma. This file only answers the
one question that was previously answered inline, twice, in two
different files: "is this a parent/child bundle or a flat record list,
and how do I turn either into Chroma-ready records?"
"""

from __future__ import annotations

from typing import Any

from codebase.vectordb.skelton import BUNDLE_KEYS, SCALAR_METADATA_TYPES, ChromaRecord


def is_bundle_payload(payload: Any) -> bool:
    """True when payload is a dict carrying both 'parents' and 'children'."""
    return isinstance(payload, dict) and all(key in payload for key in BUNDLE_KEYS)


def split_bundle(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Pull the parent and child record lists out of a bundle payload."""
    return payload.get("parents", []), payload.get("children", [])


def to_records(raw_records: list[dict[str, Any]]) -> list[ChromaRecord]:
    """Convert validated raw record dicts into ChromaRecord values."""
    return [
        ChromaRecord(id=raw["id"], text=raw["text"], metadata=raw.get("metadata") or {})
        for raw in raw_records
    ]


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only scalar metadata values that ChromaDB can store."""
    return {k: v for k, v in metadata.items() if isinstance(v, SCALAR_METADATA_TYPES)}