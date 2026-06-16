"""Shared data models for the financial RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

AnswerGenerator = Callable[[str, str, list[dict[str, Any]]], str]


@dataclass
class QueryPlan:
    """Minimal plan used by the financial RAG runner."""

    question: str
    filters: dict[str, Any] = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Structured response returned by the financial RAG runner."""

    status: str
    answer: str
    citations: list[dict[str, Any]]
    debug_json_path: str | None
    tools_used: list[dict[str, Any]]
    checkpointer: dict[str, Any]
