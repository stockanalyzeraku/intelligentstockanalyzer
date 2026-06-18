"""Rule-based query checkpointer for the RAG worker."""

from __future__ import annotations

import re

from codebase.ragrun.schemas import CheckpointResult


class RuleBasedCheckpointer:
    """Validate that a query contains company, financial year, and topic."""

    YEAR_PATTERN = re.compile(r"\b(?:FY\s*)?(20\d{2}|19\d{2}|\d{2})\b", re.IGNORECASE)
    TOPIC_WORDS = {
        "revenue", "sales", "income", "profit", "pat", "ebitda", "margin",
        "debt", "cash", "assets", "liabilities", "shareholder", "shareholding",
        "promoter", "financial", "figures", "results", "eps", "dividend",
    }
    STOPWORDS = {
        "what", "is", "are", "the", "for", "of", "in", "give", "details", "key",
        "financial", "figures", "fy", "year", "annual", "report", "and", "to",
    }

    def validate(self, query: str) -> CheckpointResult:
        normalized = " ".join((query or "").strip().split())
        if not normalized:
            return CheckpointResult(
                allowed=False,
                reason="empty_query",
                message="The query does not contain enough context.",
                missing_context=["company", "financial_year", "topic"],
            )

        financial_year = self._extract_year(normalized)
        topic = self._extract_topic(normalized)
        company = self._extract_company(normalized)

        missing = []
        if not company:
            missing.append("company")
        if not financial_year:
            missing.append("financial_year")
        if not topic:
            missing.append("topic")

        if missing:
            return CheckpointResult(
                allowed=False,
                reason="missing_required_context",
                message="The query does not contain enough context. Please include company name, financial year, and topic.",
                company=company,
                financial_year=financial_year,
                topic=topic,
                missing_context=missing,
            )

        return CheckpointResult(
            allowed=True,
            reason="accepted",
            message="Query accepted.",
            company=company,
            financial_year=financial_year,
            topic=topic,
            missing_context=[],
        )

    def _extract_year(self, query: str) -> str | None:
        match = self.YEAR_PATTERN.search(query)
        if not match:
            return None
        year = match.group(1)
        return f"FY{year}" if len(year) == 2 else year

    def _extract_topic(self, query: str) -> str | None:
        lowered = query.lower()
        for word in sorted(self.TOPIC_WORDS):
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                return word
        return None

    def _extract_company(self, query: str) -> str | None:
        query_without_year = self.YEAR_PATTERN.sub(" ", query)
        capitalized_runs = re.findall(
            r"\b[A-Z][A-Za-z&.]*\b(?:\s+\b[A-Z][A-Za-z&.]*\b)*",
            query_without_year,
        )
        candidates = []
        for run in capitalized_runs:
            words = [
                word
                for word in run.split()
                if word.lower() not in self.STOPWORDS
                and word.lower() not in self.TOPIC_WORDS
                and word.lower() != "pattern"
            ]
            if words:
                candidates.append(" ".join(words))
        if candidates:
            return max(candidates, key=len)
        tokens = [
            token
            for token in re.findall(r"[A-Za-z]+", query_without_year)
            if token.lower() not in self.STOPWORDS
            and token.lower() not in self.TOPIC_WORDS
        ]
        return " ".join(tokens[:3]) if len(tokens) >= 2 else None
