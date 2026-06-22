"""Deterministic query classification and entity extraction.

This module never calls an LLM. It exists specifically so that company/year
extraction and the financial-vs-general routing decision are NOT delegated to
the model, which removes the risk of the model silently misreading a company
name or period and then answering from its own (possibly wrong) knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Company name (lowercase, as typed by a user) -> screener/NSE symbol.
# Extend this dict as more companies are onboarded.
COMPANY_LOOKUP: dict[str, str] = {
    "kalyan jewellers": "KALYANKJIL",
    "kalyan jewellers india": "KALYANKJIL",
    "kalyankjil": "KALYANKJIL",
}

# Keywords that indicate the user wants a verified financial-table figure.
# Sourced from the line items that actually exist in the financial statements
# (Borrowings, Sales, Net Profit, EPS, etc.) plus common synonyms.
FINANCIAL_KEYWORDS: tuple[str, ...] = (
    "revenue",
    "sales",
    "profit",
    "income",
    "expense",
    "expenses",
    "eps",
    "borrowing",
    "borrowings",
    "debt",
    "reserves",
    "equity",
    "dividend",
    "cash flow",
    "cfo",
    "fcf",
    "free cash flow",
    "operating profit",
    "opm",
    "depreciation",
    "tax",
    "fixed assets",
    "total assets",
    "total liabilities",
    "investments",
    "cwip",
)

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


@dataclass
class ClassificationResult:
    """Result of deterministic query classification.

    Attributes:
        intent: Either "financial" or "general".
        symbol: Resolved company symbol, or None if not recognized.
        period: Resolved period string like "Mar 2023", or None if not found.
        raw_year: The raw 4-digit year extracted from the query, if any.
        unresolved_reason: Human-readable reason extraction failed, if any.
    """

    intent: str
    symbol: str | None
    period: str | None
    raw_year: str | None
    unresolved_reason: str | None = None


def _extract_symbol(query: str) -> str | None:
    """Find a known company name in the query and return its symbol.

    Args:
        query: The raw user query.

    Returns:
        The resolved symbol, or None if no known company name matched.
    """
    lowered = query.lower()
    for name, symbol in COMPANY_LOOKUP.items():
        if name in lowered:
            return symbol
    return None


def _extract_year(query: str) -> str | None:
    """Find a 4-digit year in the query.

    Args:
        query: The raw user query.

    Returns:
        The matched year as a string (e.g. "2023"), or None if not found.
    """
    match = YEAR_PATTERN.search(query)
    return match.group(1) if match else None

def _classify_financial_intent(query: str) -> bool:
    """Decide whether the query is asking for a verified financial figure.

    Args:
        query: The raw user query.

    Returns:
        True if any known financial keyword appears in the query.
    """
    lowered = query.lower()
    return any(keyword in lowered for keyword in FINANCIAL_KEYWORDS)


def classify_intent(query: str) -> ClassificationResult:
    """Classify a user query and extract company/year, deterministically.

    Args:
        query: The raw user query, e.g. "What was revenue of Kalyan
            Jewellers in 2023".

    Returns:
        A ClassificationResult describing intent and extracted entities.
        If company or year can't be resolved, `unresolved_reason` is set so
        the caller can short-circuit before invoking any LLM.
    """
    symbol = _extract_symbol(query)
    raw_year = _extract_year(query)
    period = f"Mar {raw_year}" if raw_year else None
    is_financial = _classify_financial_intent(query)
    intent = "financial" if is_financial else "general"

    unresolved_reason = None
    if symbol is None:
        unresolved_reason = (
            "I couldn't identify a known company in your question. "
            "Please mention the company name explicitly."
        )
    elif is_financial and period is None:
        unresolved_reason = (
            "I couldn't identify a year/period in your question. "
            "Please mention a specific year (e.g. 2023)."
        )

    logger.info(
        "Classified query=%r -> intent=%s symbol=%s period=%s",
        query,
        intent,
        symbol,
        period,
    )

    return ClassificationResult(
        intent=intent,
        symbol=symbol,
        period=period,
        raw_year=raw_year,
        unresolved_reason=unresolved_reason,
    )
