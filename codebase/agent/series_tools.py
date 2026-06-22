"""Tools for the Stage 4 Data Retrieval Agent (multi-agent pipeline).

These are NEW tools, added alongside the existing single-period
get_financial_data in codebase/agent/tools.py (which is left untouched).
The difference: get_financial_data fetches one (symbol, line_item, period)
triple per call. A 3-year comparison needs 3 of those, which an agent has
to remember to issue correctly. get_financial_series fetches ALL requested
periods for a line_item in one call, against the typed financials tables
built in codebase/financials/, via the existing discovery.py API rather
than hand-rolled SQL.

Both tools here return small JSON-serializable dicts/strings (never raise)
so the calling agent's tool-loop never crashes on a bad lookup - same
defensive style as the existing tools.py.
"""

from __future__ import annotations

import logging

from codebase.financials import discovery

logger = logging.getLogger(__name__)

# Guarded import: BaseModel/Field are only needed to build the @tool args
# schemas below. The plain fetch_financial_series()/list-items logic has
# no pydantic dependency at all and stays directly unit-testable without
# it installed.
try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised in environments without pydantic
    class BaseModel:  # type: ignore[no-redef]
        """Minimal fallback so module import doesn't fail without pydantic."""

    def Field(*_args, default=None, **_kwargs):  # type: ignore[no-redef]
        return default


class FinancialSeriesInput(BaseModel):
    """Input schema for the get_financial_series tool."""

    symbol: str = Field(
        ...,
        description=(
            "Company ticker/screener symbol, e.g. 'KALYANKJIL'. Must be "
            "resolved ahead of time by the Query Understanding stage, "
            "never guessed by this agent."
        ),
    )
    line_item: str = Field(
        ...,
        description=(
            "Exact financial line item name, e.g. 'Sales', 'Net Profit', "
            "'EPS in Rs', 'Borrowings'. If the exact label is unknown, call "
            "list_available_line_items first to find the correct spelling."
        ),
    )
    periods: list[str] = Field(
        ...,
        description=(
            "Period labels to fetch, e.g. ['Mar 2021', 'Mar 2022', 'Mar 2023']. "
            "Fetch ALL periods needed for this line_item in a single call - "
            "do not call this tool once per period."
        ),
    )


class ListLineItemsInput(BaseModel):
    """Input schema for the list_available_line_items tool."""

    symbol: str = Field(..., description="Company ticker/screener symbol, e.g. 'KALYANKJIL'.")


# Statement tables to search, in a fixed order, when looking for a
# line_item without knowing which statement it belongs to. Mirrors the
# search order in codebase/agent/tools.py's _STATEMENT_TABLES.
_STATEMENT_TABLE_ORDER: tuple[str, ...] = (
    "statement_profit_loss",
    "statement_balance_sheet",
    "statement_cash_flow",
)


def _resolve_company_id(symbol: str) -> int | None:
    """Look up a company_id for a symbol via discovery.find_company.

    Returns None if the symbol isn't found in the financials database at
    all (distinct from "found, but this line_item/period is missing").
    """
    company = discovery.find_company(symbol)
    return company["company_id"] if company else None


def _find_table_for_line_item(symbol_company_id: int, line_item: str) -> str | None:
    """Find which statement_* table actually contains this line_item for
    this company, searching in a fixed order. Returns the table name, or
    None if the line_item isn't found in any statement for this company.
    """
    for table_name in _STATEMENT_TABLE_ORDER:
        items = discovery.list_line_items(table_name, company_id=symbol_company_id)
        if line_item in items:
            return table_name
    return None


def fetch_financial_series(symbol: str, line_item: str, periods: list[str]) -> dict:
    """Plain Python function (no @tool wrapper) doing the actual lookup.

    Exposed separately from the @tool-wrapped version below so the
    deterministic delta-computation step (run right after Stage 4, before
    Stage 6 ever sees the data) can call this directly without going
    through the LangChain tool-calling machinery.

    Returns a dict shaped either:
        {"ok": True, "symbol": ..., "line_item": ..., "unit": ...,
         "values": {"Mar 2021": 123.4, "Mar 2022": None, ...},
         "table": "statement_profit_loss"}
    or:
        {"ok": False, "error": "<human-readable reason>"}

    A period present in `periods` but with no recorded value comes back as
    None in `values` (never silently dropped), so downstream code/prompts
    can distinguish "not disclosed for this period" from "never asked for".
    """
    company_id = _resolve_company_id(symbol)
    if company_id is None:
        return {"ok": False, "error": f"Company '{symbol}' not found in financials database."}

    table_name = _find_table_for_line_item(company_id, line_item)
    if table_name is None:
        return {
            "ok": False,
            "error": (
                f"Line item '{line_item}' not found for '{symbol}' in any "
                f"statement table. Call list_available_line_items to see "
                f"valid names."
            ),
        }

    rows = discovery.get_statement(
        table_name, company_id, line_items=[line_item], periods=periods, pivot=False
    )

    values: dict[str, float | None] = {p: None for p in periods}
    unit = None
    for row in rows:
        values[row["period_label"]] = row["value"]
        unit = row["unit"]

    return {
        "ok": True,
        "symbol": symbol,
        "line_item": line_item,
        "unit": unit,
        "values": values,
        "table": table_name,
    }


# --- LangChain @tool wrappers -------------------------------------------
# Guarded import: the plain fetch_financial_series()/list-items logic above
# has no LangChain dependency and is directly unit-testable without it
# installed. The @tool-wrapped versions below require langchain and are
# what actually gets handed to the Stage 4 agent at runtime.
try:
    from langchain.tools import tool
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without langchain
    _LANGCHAIN_AVAILABLE = False

    def tool(*_args, **_kwargs):  # type: ignore[no-redef]
        """No-op fallback decorator so module import doesn't fail."""
        def _decorator(func):
            return func
        return _decorator


@tool("get_financial_series", args_schema=FinancialSeriesInput)
def get_financial_series(symbol: str, line_item: str, periods: list[str]) -> str:
    """Fetch a verified financial line item across MULTIPLE periods in one call.

    This is the only source of truth for quantitative financial figures in
    the multi-agent pipeline. Always fetch every period needed for the
    current line_item in a single call rather than calling once per period.

    Args:
        symbol: Company ticker/screener symbol.
        line_item: Exact financial line item name.
        periods: List of period labels to fetch, e.g. ["Mar 2021", "Mar 2022", "Mar 2023"].

    Returns:
        A short summary string. Use list_available_line_items first if you
        are not sure of the exact line_item spelling for this company.
    """
    result = fetch_financial_series(symbol, line_item, periods)
    if not result["ok"]:
        return f"ERROR: {result['error']}"

    parts = [f"{p}: {v if v is not None else 'not disclosed'}" for p, v in result["values"].items()]
    unit_suffix = f" ({result['unit']})" if result["unit"] else ""
    return f"{line_item} for {symbol}{unit_suffix}: " + "; ".join(parts)


@tool("list_available_line_items", args_schema=ListLineItemsInput)
def list_available_line_items(symbol: str) -> str:
    """List every financial line item available for a company, across all statements.

    Use this when unsure of the exact line_item spelling/name before
    calling get_financial_series, or to discover what data exists at all.

    Args:
        symbol: Company ticker/screener symbol.

    Returns:
        A string grouping line items by statement table, or an "ERROR: ..."
        string if the company isn't found.
    """
    company_id = _resolve_company_id(symbol)
    if company_id is None:
        return f"ERROR: Company '{symbol}' not found in financials database."

    sections = []
    for table_name in _STATEMENT_TABLE_ORDER:
        items = discovery.list_line_items(table_name, company_id=company_id)
        if items:
            sections.append(f"{table_name}: {', '.join(items)}")
    return "\n".join(sections) if sections else f"No line items found for '{symbol}'."
