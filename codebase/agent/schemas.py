"""Structured-output contract for the Query Understanding stage.

This schema is what the Query Understanding LLM call is constrained to
produce (via structured output / tool-call forcing, not free text). Code
immediately downstream validates and acts on these fields - the LLM never
gets to silently decide anything past this point; it only extracts.

Design intent (see conversation/design doc for full rationale):
  - line_items are ONLY what the user named or clearly implied. The
    pipeline does NOT add unrelated metrics on its own (e.g. asking about
    Sales does not pull in margins or borrowings).
  - periods reflect what was asked. If the user gave a single year with no
    explicit "just this year" / "only" qualifier, downstream code expands
    this to a trailing 3-year window by default (per product decision) -
    that expansion is deterministic code, not something the LLM decides.
  - needs_qualitative_context flags whether annual-report search (Stage 5)
    should run at all - it should be true only for "why / strategy / driven
    by / outlook / risk" style questions, false for pure number lookups.
  - comparison_requested is true when the user explicitly asked to compare
    across periods (e.g. "compare 2021 vs 2023", "trend over 3 years").
  - ambiguity_reason is set ONLY when the company itself cannot be
    resolved. Year ambiguity is no longer a stop condition (resolved by the
    trailing-3-year default instead), per product decision.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    """Structured extraction result for one user query."""

    symbol: str | None = Field(
        default=None,
        description=(
            "Resolved company ticker/screener symbol, e.g. 'KALYANKJIL'. "
            "None if no company could be identified in the query."
        ),
    )
    company_name_as_given: str | None = Field(
        default=None,
        description="The company name/phrase as the user wrote it, for logging/debugging.",
    )
    line_items: list[str] = Field(
        default_factory=list,
        description=(
            "Exact financial line item name(s) the user asked about, e.g. "
            "['Sales'], ['Net Profit', 'EPS in Rs']. ONLY what was named or "
            "clearly implied - never add related metrics the user didn't ask for. "
            "Use exact line item names as they appear in the financial statements "
            "(Sales, Net Profit, EPS in Rs, Borrowings, Reserves, Operating Profit, "
            "Cash from Operating Activity, etc.) - if unsure of the exact label, "
            "use the closest common synonym and let downstream lookup resolve it. "
            "Do NOT put computed ratios here (e.g. ROE, Net Profit Margin, "
            "Debt-to-Equity) - those go in derived_metrics instead. The one "
            "exception is 'OPM %' (Operating Margin), which IS a directly "
            "published line item, not computed - use line_items for it."
        ),
    )
    derived_metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Computed/ratio metrics the user asked about that are NOT directly "
            "published line items, e.g. ['ROE'], ['Net Profit Margin', "
            "'Debt to Equity']. Recognized values: 'Net Profit Margin', 'ROE', "
            "'Debt to Equity', 'Free Cash Flow Margin'. These are calculated "
            "from raw line items rather than read directly from the source, "
            "and will always be presented as approximate/computed figures. "
            "Do NOT put 'Operating Margin'/'OPM' here - that is a directly "
            "published line item, use line_items for it instead."
        ),
    )
    raw_years: list[str] = Field(
        default_factory=list,
        description=(
            "4-digit fiscal years explicitly mentioned by the user, e.g. ['2023'] "
            "or ['2021', '2023']. Empty if no year was mentioned at all (downstream "
            "code will default to the most recent available year + trailing context)."
        ),
    )
    comparison_requested: bool = Field(
        default=False,
        description=(
            "True if the user explicitly asked to compare across periods or see a "
            "trend (e.g. 'compare', 'trend', 'over the years', 'vs', 'growth over'). "
            "False for a single-point-in-time question."
        ),
    )
    single_year_only: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly restricted scope to one year and "
            "signaled they do NOT want surrounding context, e.g. 'just give me "
            "2023, nothing else' or 'only FY2023, no comparison'. In the ordinary "
            "case of a plain single-year question with no such qualifier, leave "
            "this False so downstream code applies the trailing-3-year default."
        ),
    )
    needs_qualitative_context: bool = Field(
        default=False,
        description=(
            "True if the question asks for explanation/reasoning/narrative - "
            "words like 'why', 'reason', 'driven by', 'strategy', 'outlook', "
            "'risk', 'management commentary'. False for pure number lookups."
        ),
    )
    intent: str = Field(
        default="financial",
        description=(
            "'financial' if the question is fundamentally about a verified numeric "
            "figure (even if it also wants qualitative context alongside it), "
            "'general' if it is purely qualitative/narrative with no specific "
            "number requested at all."
        ),
    )
    ambiguity_reason: str | None = Field(
        default=None,
        description=(
            "Set ONLY if the company could not be identified/resolved at all. "
            "A short, user-facing clarification message, e.g. 'I could not "
            "identify a known company in your question.' Leave None if a company "
            "was resolved, even if other fields (years, line_items) are empty - "
            "those have safe defaults downstream."
        ),
    )
