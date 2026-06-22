"""Stage 7: Follow-up Suggestor.

Deliberately NOT an LLM call. Per the narrowed scope of "intelligent"
(only synthesize what was asked, no proactive cross-metric speculation),
suggesting next questions is a different kind of task than answering the
current one: it only needs to know what data EXISTS but wasn't used, which
is answerable directly from the database schema via
codebase.financials.discovery - no model judgment, no risk of suggesting
a metric that doesn't actually exist.
"""

from __future__ import annotations

from codebase.financials import discovery

MAX_SUGGESTIONS = 3

# A handful of "interesting" line items per statement table to prioritize
# when suggesting OTHER metrics - keeps suggestions useful rather than
# offering up obscure rows first. Falls back to whatever else exists if
# none of these are available.
_PRIORITY_LINE_ITEMS: tuple[str, ...] = (
    "Net Profit",
    "Sales",
    "EPS in Rs",
    "Operating Profit",
    "Cash from Operating Activity",
    "Borrowings",
    "Reserves",
)


def _other_available_line_items(symbol: str, already_asked: list[str]) -> list[str]:
    """Line items that exist for this company but weren't part of this query."""
    company = discovery.find_company(symbol)
    if not company:
        return []

    already_asked_lower = {li.lower() for li in already_asked}
    found: list[str] = []
    for table_name in discovery.list_statement_tables():
        for item in discovery.list_line_items(table_name, company_id=company["company_id"]):
            if item.lower() not in already_asked_lower and item not in found:
                found.append(item)

    # Prioritize well-known/interesting metrics first, then whatever's left.
    prioritized = [li for li in _PRIORITY_LINE_ITEMS if li in found]
    rest = [li for li in found if li not in prioritized]
    return prioritized + rest


def _other_available_periods(symbol: str, already_used: list[str]) -> list[str]:
    """Periods that exist for this company but weren't part of this query."""
    company = discovery.find_company(symbol)
    if not company:
        return []

    already_used_set = set(already_used)
    periods: set[str] = set()
    for table_name in discovery.list_statement_tables():
        rows = discovery.get_statement(table_name, company["company_id"], pivot=False)
        for row in rows:
            if row["period_label"] not in already_used_set:
                periods.add(row["period_label"])

    def _year_of(label: str) -> int:
        parts = label.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    return sorted(periods, key=_year_of)


def suggest_follow_ups(
    symbol: str,
    line_items_asked: list[str],
    periods_used: list[str],
    qualitative_context_was_used: bool,
) -> list[str]:
    """Generate up to MAX_SUGGESTIONS follow-up question strings.

    Purely template-based over real, verified schema/data facts - never
    invents a metric or period that doesn't exist for this company.
    Returns [] if the company has no data at all, rather than falling
    through to a generic suggestion that isn't actually backed by data.
    """
    if discovery.find_company(symbol) is None:
        return []

    suggestions: list[str] = []

    other_line_items = _other_available_line_items(symbol, line_items_asked)
    if other_line_items:
        next_metric = other_line_items[0]
        suggestions.append(f"Want to see {next_metric} for the same period?")

    other_periods = _other_available_periods(symbol, periods_used)
    if other_periods:
        earliest_other = other_periods[0]
        suggestions.append(f"Compare this against {earliest_other} as well?")

    if not qualitative_context_was_used and line_items_asked:
        item_phrase = line_items_asked[0] if len(line_items_asked) == 1 else "these figures"
        suggestions.append(f"Curious what drove the change in {item_phrase}? I can search the annual report.")

    if len(other_line_items) > 1 and len(suggestions) < MAX_SUGGESTIONS:
        second_metric = other_line_items[1]
        suggestions.append(f"See {second_metric} trend over the last few years?")

    return suggestions[:MAX_SUGGESTIONS]
