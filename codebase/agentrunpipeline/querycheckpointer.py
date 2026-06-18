"""Basic query validation before running expensive retrieval."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from typing import Any


class QueryCheckpointer:
    """Simple guardrail that asks for more context when a query is too basic."""

    BASIC_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"^what\s+is\s+(revenue|profit|pat|ebitda|debt|sales|income)\??$", re.IGNORECASE),
        re.compile(r"^(revenue|profit|pat|ebitda|debt|sales|income)\??$", re.IGNORECASE),
    )

    def validate(
        self,
        question: str,
        company: str | None = None,
        year: int | str | None = None,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        """Return a basic accept/reject decision for the user question."""
        normalized = " ".join(question.strip().split())
        missing_context = [name for name, value in (("company", company), ("year", year)) if not value]
        matched_basic_pattern = any(pattern.match(normalized) for pattern in self.BASIC_PATTERNS)

        if not normalized:
            return {
                "allowed": False,
                "reason": "empty_question",
                "message": "Please provide a financial question with company and year.",
                "missing_context": ["question", *missing_context],
            }

        if matched_basic_pattern and missing_context:
            return {
                "allowed": False,
                "reason": "query_too_basic",
                "message": "Need to provide more information: include at least company and year.",
                "missing_context": missing_context,
            }

        return {
            "allowed": True,
            "reason": "accepted",
            "message": "Query accepted.",
            "missing_context": missing_context,
            "doc_type": doc_type,
        }
