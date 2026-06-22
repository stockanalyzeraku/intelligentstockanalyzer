"""Stage 2: Clarification Gate (deterministic, NOT an LLM call).

Takes the QueryUnderstanding output from Stage 1 and decides:
  1. Can we proceed at all? (only blocks if the company itself is unresolved)
  2. What periods should actually be fetched, given the trailing-3-year
     default and what data actually exists for this company?

This is plain code on purpose - the company-resolution decision and the
default-window expansion must be auditable and reproducible, never a
judgment call left to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codebase.agent.schemas import QueryUnderstanding
from codebase.financials import discovery

DEFAULT_TRAILING_YEARS = 3


@dataclass
class ResolvedQuery:
    """Output of the clarification gate: either a stop signal, or a fully
    resolved set of periods ready for Stage 4 to fetch.
    """

    can_proceed: bool
    symbol: str | None = None
    line_items: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)  # chronological order, oldest first
    comparison_requested: bool = False
    needs_qualitative_context: bool = False
    intent: str = "financial"
    stop_message: str | None = None  # set when can_proceed is False


def _period_label(year: str) -> str:
    """'2023' -> 'Mar 2023'. Matches the period_label convention used
    throughout codebase/financials (annual periods are always fiscal-year-
    end labels like 'Mar 2023')."""
    return f"Mar {year}"


def _available_periods_for_company(symbol: str) -> list[str]:
    """Return every period_label that exists for this company, across any
    statement table, sorted chronologically (oldest first).

    Uses _meta_line_items-independent data: queries the per-company line
    items directly so this works even before knowing which line_items the
    user asked about - we just need to know what YEARS exist at all.
    """
    company = discovery.find_company(symbol)
    if not company:
        return []

    periods: set[str] = set()
    for table_name in discovery.list_statement_tables():
        rows = discovery.get_statement(table_name, company["company_id"], pivot=False)
        for row in rows:
            periods.add(row["period_label"])

    # Period labels are "Mon YYYY" - sort by the year component.
    def _year_of(period_label: str) -> int:
        parts = period_label.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    return sorted(periods, key=_year_of)


def _resolve_periods(
    raw_years: list[str],
    single_year_only: bool,
    available_periods: list[str],
) -> list[str]:
    """Turn extracted raw_years into a final, chronologically-ordered list
    of period labels to fetch, applying the trailing-3-year default.

    Rules:
      - No years mentioned at all -> trailing DEFAULT_TRAILING_YEARS most
        recent available periods for this company (the product default).
      - One year mentioned, single_year_only=True -> exactly that one
        period (if it exists; if not, falls back to the closest available
        period rather than returning nothing).
      - One year mentioned, single_year_only=False (the common case) ->
        that year PLUS the DEFAULT_TRAILING_YEARS-1 periods before it, per
        the "default to 3 years" product decision - context is added
        automatically, not asked for.
      - Multiple years mentioned -> exactly those periods, in chronological
        order (the user was explicit; we don't add more).
    """
    if not available_periods:
        return []

    requested_labels = [_period_label(y) for y in raw_years]

    if not requested_labels:
        return available_periods[-DEFAULT_TRAILING_YEARS:]

    if len(requested_labels) == 1:
        target = requested_labels[0]
        if single_year_only:
            return [target] if target in available_periods else available_periods[-1:]
        if target in available_periods:
            idx = available_periods.index(target)
            window_start = max(0, idx - (DEFAULT_TRAILING_YEARS - 1))
            return available_periods[window_start : idx + 1]
        # Requested year isn't in the data at all - fall back to the
        # trailing default window so the user still gets useful context
        # rather than an empty result.
        return available_periods[-DEFAULT_TRAILING_YEARS:]

    # Multiple years explicitly given - use exactly what exists among them,
    # in chronological order. Silently dropping a requested-but-nonexistent
    # year here is intentional: Stage 4/enrichment already represent a
    # missing period as None per-line-item, which is the correct place to
    # surface "not disclosed", not here.
    return [p for p in available_periods if p in requested_labels]


def resolve_query(understanding: QueryUnderstanding) -> ResolvedQuery:
    """Run the clarification gate on a Stage 1 QueryUnderstanding result."""

    if understanding.symbol is None or understanding.ambiguity_reason is not None:
        return ResolvedQuery(
            can_proceed=False,
            stop_message=understanding.ambiguity_reason
            or "I couldn't identify a known company in your question. Please mention the company name explicitly.",
        )

    available_periods = _available_periods_for_company(understanding.symbol)
    if not available_periods:
        return ResolvedQuery(
            can_proceed=False,
            stop_message=(
                f"I don't have any financial data stored for '{understanding.symbol}' yet."
            ),
        )

    periods = _resolve_periods(
        understanding.raw_years, understanding.single_year_only, available_periods
    )

    return ResolvedQuery(
        can_proceed=True,
        symbol=understanding.symbol,
        line_items=understanding.line_items,
        periods=periods,
        comparison_requested=understanding.comparison_requested,
        needs_qualitative_context=understanding.needs_qualitative_context,
        intent=understanding.intent,
    )
