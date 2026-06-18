# """Shared data models and LLM adapters for the financial RAG pipeline."""

# from __future__ import annotations
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from dataclasses import dataclass, field
# from typing import Any, Callable, Protocol, runtime_checkable

# AnswerGenerator = Callable[[str, str, list[dict[str, Any]]], str]


# @runtime_checkable
# class AnswerProvider(Protocol):
#     """Provider interface for answer-generation backends."""

#     name: str

#     def generate(self, question: str, context: str, records: list[dict[str, Any]]) -> str:
#         """Generate an answer from the question and retrieved context."""


# @dataclass
# class QueryPlan:
#     """Minimal plan used by the financial RAG runner."""

#     question: str
#     filters: dict[str, Any] = field(default_factory=dict)
#     expanded_queries: list[str] = field(default_factory=list)


# @dataclass
# class LLMModelConfig:
#     """Configuration for one answer-generation model."""

#     provider: str
#     model: str
#     api_key: str | None = None
#     temperature: float = 0.0
#     max_output_tokens: int | None = None
#     timeout_seconds: int | None = None


# @dataclass
# class RAGResponse:
#     """Structured response returned by the financial RAG runner."""

#     status: str
#     answer: str
#     citations: list[dict[str, Any]]
#     debug_json_path: str | None
#     tools_used: list[dict[str, Any]]
#     checkpointer: dict[str, Any]
#     cache: dict[str, Any] = field(default_factory=dict)
