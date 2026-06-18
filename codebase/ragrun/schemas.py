"""Typed request, response, and debug objects for the RAG worker."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CheckpointResult:
    """Rule-based validation result for a user query."""

    allowed: bool
    reason: str
    message: str
    company: str | None = None
    financial_year: str | None = None
    topic: str | None = None
    missing_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    """One chunk selected from Chroma as answer evidence."""

    chunk_id: str | None
    parent_id: str | None
    text: str
    page_number: int | str | None
    source: str | None
    company: str | None
    financial_year: str | None
    distance: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRunResult:
    """Outcome of asking one or more LLMs for an answer."""

    answer: str
    model_used: str | None
    provider_used: str | None
    fallback_used: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGWorkerResponse:
    """Final response returned by the background RAG worker."""

    status: str
    answer: str
    source: str
    debug_json_path: str | None
    checkpointer: dict[str, Any]
    cache: dict[str, Any]
    model: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
