"""
Readable JSON citation/debug traces for RAG answers.

The RAG pipeline writes one JSON file per answer so developers can inspect the
question, checkpointer decision, retrieval tools used, source chunks/pages, and
final answer text without reading logs.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ToolTrace:
    """One tool invocation performed while answering a query."""

    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A source record used as evidence for an answer."""

    source_id: str
    parent_id: str | None
    child_id: str | None
    page_number: int | str | None
    company: str | None
    report_year: int | str | None
    doc_type: str | None
    page_intent: str | None
    distance: float | None
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGDebugTrace:
    """Complete JSON-serialisable trace for a RAG answer."""

    trace_id: str
    created_at: str
    question: str
    status: str
    answer: str
    checkpointer: dict[str, Any]
    filters: dict[str, Any]
    expanded_queries: list[str]
    tools_used: list[ToolTrace]
    citations: list[Citation]


class CitationDebugWriter:
    """Build and persist clean JSON traces for financial RAG answers."""

    def __init__(self, output_dir: str | Path = "rag_debug") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_trace(
        self,
        question: str,
        status: str,
        answer: str,
        checkpointer: dict[str, Any],
        filters: dict[str, Any],
        expanded_queries: list[str],
        tools_used: list[ToolTrace],
        citations: list[Citation],
    ) -> RAGDebugTrace:
        """Create an in-memory trace object for one pipeline run."""
        return RAGDebugTrace(
            trace_id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            question=question,
            status=status,
            answer=answer,
            checkpointer=checkpointer,
            filters=filters,
            expanded_queries=expanded_queries,
            tools_used=tools_used,
            citations=citations,
        )

    def write_trace(self, trace: RAGDebugTrace) -> Path:
        """Write a trace as readable JSON and return the output path."""
        filename = f"{self._slugify(trace.question)}_{trace.trace_id[:8]}.json"
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(trace), fh, ensure_ascii=False, indent=2, default=str)
        return path

    @staticmethod
    def _slugify(value: str, max_length: int = 60) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return (slug or "rag_trace")[:max_length]
