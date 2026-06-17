"""Reusable fake Chroma store for financial pipeline tests."""

from __future__ import annotations

from typing import Any


class FakeChromaStore:
    """Small fake that mimics ChromaStore.query_children_with_parent_context."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.records = records or [self.default_record()]

    def query_children_with_parent_context(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        self.requests.append({"query_texts": query_texts, "n_results": n_results, "where": where})
        return self.records

    @staticmethod
    def default_record() -> dict[str, Any]:
        return {
            "id": "page_10_parent",
            "text": "Revenue was Rs 100 crore in FY2025.",
            "metadata": {
                "company": "TEST",
                "year": 2025,
                "page_number": 10,
                "page_intent": "financial_highlights",
            },
            "parent_id": "page_10_parent",
            "parent_text": "Revenue was Rs 100 crore in FY2025. PAT was Rs 10 crore.",
            "parent_metadata": {
                "company": "TEST",
                "year": 2025,
                "page_number": 10,
                "page_intent": "financial_highlights",
            },
            "child_id": "page_10_parent_child_0",
            "child_text": "Revenue was Rs 100 crore in FY2025.",
            "child_metadata": {"company": "TEST", "year": 2025, "page_number": 10},
            "distance": 0.12,
        }
