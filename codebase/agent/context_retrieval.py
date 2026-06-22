"""Stage 5: Context Retrieval Agent.

Runs ONLY when Stage 1/2 set needs_qualitative_context=True. Wraps the
EXISTING search_annual_report tool (codebase/agent/tools.py, unchanged)
rather than reimplementing vector search.

Scoping constraint (traced from the real ChromaStore.query_collection /
query_children_with_parent_context implementation): the `where` filter
supports at most ONE year at a time (where["year"] = int(year)) - there is
no multi-year filter in the underlying tool. So a 3-year qualitative
question requires one search_annual_report call PER PERIOD, not one call
covering a range. This module makes that explicit and bounded rather than
letting an LLM agent decide how many times to call the tool.

This stage produces grounded TEXT SNIPPETS ONLY - it never produces a
number. It is also explicitly scoped to the line_items/periods already
verified by Stage 4: it does not go searching for new metrics the user
didn't ask about.
"""

from __future__ import annotations

import logging

from codebase.agent.tools import search_annual_report

logger = logging.getLogger(__name__)

# Cap on how many of the resolved periods we'll search the annual report
# for. A query with many periods (e.g. "5 year trend") would otherwise
# trigger one search_annual_report call per period - this keeps API usage
# bounded even though API calls are not currently a hard constraint.
MAX_PERIODS_TO_SEARCH = 3


def _invoke_search_tool(symbol: str, query: str, year: str | None, n_results: int) -> str:
    """Call the @tool-wrapped search_annual_report via .invoke(), the
    correct calling convention for a LangChain StructuredTool (a single
    dict of named args - NOT positional call syntax).
    """
    return search_annual_report.invoke(
        {"symbol": symbol, "query": query, "year": year, "n_results": n_results}
    )


def _year_from_period(period_label: str) -> str | None:
    """'Mar 2023' -> '2023'. Returns None if the label isn't in this format."""
    parts = period_label.strip().split()
    return parts[-1] if len(parts) == 2 and parts[-1].isdigit() else None


def retrieve_context(
    symbol: str,
    line_items: list[str],
    periods: list[str],
    original_query: str,
    n_results: int = 3,
) -> dict:
    """Fetch grounded annual-report context for the resolved query scope.

    Builds one search query per (line_item, period) the resolved query
    actually covers - bounded by MAX_PERIODS_TO_SEARCH - rather than a
    single unscoped search, so the retrieved text stays anchored to what
    was actually asked about and verified in Stage 4.

    Parameters
    ----------
    symbol : str
        Resolved company symbol.
    line_items : list[str]
        The line items already verified by Stage 4 (e.g. ["Sales"]).
    periods : list[str]
        The periods already resolved by Stage 2 (e.g. ["Mar 2023", "Mar 2024", "Mar 2025"]).
    original_query : str
        The user's raw question - used to focus the semantic search
        (e.g. "why did sales grow"), not just the line item name alone.
    n_results : int
        Max snippets per individual search call.

    Returns
    -------
    dict
        {
          "ok": bool,
          "snippets": [{"period": "Mar 2023", "line_item": "Sales", "text": "...", "distance": 0.12}, ...],
          "searched_periods": [...],   # which periods were actually searched
          "skipped_periods": [...],    # periods dropped due to MAX_PERIODS_TO_SEARCH
          "errors": [...],             # any "ERROR: ..." strings the tool returned, kept for visibility
        }
        snippets is [] (not an error) when nothing was found - that is a
        legitimate outcome (e.g. no annual report indexed for this
        company), and callers (Stage 6) must treat it as "no qualitative
        context available", not as a failure.
    """
    periods_to_search = periods[-MAX_PERIODS_TO_SEARCH:]
    skipped_periods = periods[: len(periods) - len(periods_to_search)]

    snippets: list[dict] = []
    errors: list[str] = []

    for line_item in line_items:
        for period in periods_to_search:
            year = _year_from_period(period)
            query_text = f"{original_query} ({line_item}, {period})"

            try:
                raw_result = _invoke_search_tool(symbol, query_text, year, n_results)
            except Exception:  # noqa: BLE001 - this stage must never crash the pipeline
                logger.exception(
                    "search_annual_report tool call failed for symbol=%s line_item=%s period=%s",
                    symbol, line_item, period,
                )
                errors.append(f"ERROR: search_annual_report call failed for {line_item} / {period}")
                continue

            if isinstance(raw_result, str) and raw_result.startswith("ERROR:"):
                errors.append(raw_result)
                continue

            # search_annual_report returns sections joined by "\n\n---\n\n"
            # (see tools.py) - split back out so each snippet can be
            # attributed to its (line_item, period) scope individually.
            for section in str(raw_result).split("\n\n---\n\n"):
                section = section.strip()
                if section:
                    snippets.append({"period": period, "line_item": line_item, "text": section})

    return {
        "ok": True,
        "snippets": snippets,
        "searched_periods": periods_to_search,
        "skipped_periods": skipped_periods,
        "errors": errors,
    }
