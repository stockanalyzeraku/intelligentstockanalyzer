"""Agent tools: verified financial data (SQL) and annual-report search (vector).

Both tools catch their specific exceptions and return an "ERROR: ..." string
rather than raising, so the agent loop never crashes on a bad lookup and the
model can react to the failure in its final answer.
"""

from __future__ import annotations

import logging
import sqlite3

from langchain.tools import tool
from pydantic import BaseModel, Field

from codebase.financials.db import get_connection
from codebase.vectordb.chromastore import ChromaStore

logger = logging.getLogger(__name__)

_STATEMENT_TABLES: tuple[str, ...] = (
    "statement_profit_loss",
    "statement_balance_sheet",
    "statement_cash_flow",
)


class FinancialDataInput(BaseModel):
    symbol: str = Field(
        ...,
        description=(
            "Company ticker/screener symbol, e.g. 'KALYANKJIL'. Must be "
            "resolved ahead of time, never guessed by the model."
        ),
    )
    line_item: str = Field(
        ...,
        description=(
            "Exact financial line item name, e.g. 'Sales', 'Net Profit', "
            "'EPS in Rs', 'Borrowings'."
        ),
    )
    period: str = Field(
        ..., description="Period column to read, e.g. 'Mar 2023'."
    )


class AnnualReportSearchInput(BaseModel):
    symbol: str = Field(..., description="Company symbol, e.g. 'KALYANKJIL'.")
    query: str = Field(
        ..., description="Natural-language question to search the annual report for."
    )
    year: str | None = Field(
        default=None,
        description="4-digit annual report year to filter to, e.g. '2023'. Omit to search all years.",
    )
    n_results: int = Field(
        default=3, description="Number of matching report sections to retrieve."
    )


def _find_line_item_value(
    cursor: sqlite3.Cursor, symbol: str, line_item: str, period: str
) -> tuple[str, float | None, str | None] | None:
    """Search statement_* tables for a (company, line_item) row and read a period.

    Returns:
        A tuple of (table_name, value, unit) if found, else None.
    """
    cursor.execute(
        "SELECT id FROM companies WHERE screener_symbol = ? OR nse_symbol = ? LIMIT 1",
        (symbol, symbol),
    )
    company_row = cursor.fetchone()
    if company_row is None:
        return None
    company_id = company_row[0]

    for table in _STATEMENT_TABLES:
        query = (
            f'SELECT "{period}", unit FROM {table} '
            "WHERE company_id = ? AND line_item = ? LIMIT 1"
        )
        try:
            cursor.execute(query, (company_id, line_item))
        except sqlite3.OperationalError:
            continue
        row = cursor.fetchone()
        if row is not None:
            return table, row[0], row[1]
    return None


@tool("get_financial_data", args_schema=FinancialDataInput)
def get_financial_data(symbol: str, line_item: str, period: str) -> str:
    """Fetch a single verified financial line item for a company and period.

    This is the only source of truth for quantitative financial figures.
    The model must never state a number that did not come from this tool.

    Args:
        symbol: Company ticker/screener symbol.
        line_item: Exact financial line item name.
        period: Period column to read, e.g. "Mar 2023".

    Returns:
        A short string with the value and unit, or an "ERROR: ..." string
        if the company, line item, or period could not be found.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            result = _find_line_item_value(cursor, symbol, line_item, period)
    except sqlite3.Error as exc:
        logger.exception("DB error fetching %s/%s/%s", symbol, line_item, period)
        return f"ERROR: database error fetching {line_item} for {symbol} ({period}): {exc}"

    if result is None:
        return (
            f"ERROR: no data found for symbol={symbol!r}, "
            f"line_item={line_item!r}, period={period!r}."
        )

    table, value, unit = result
    if value is None:
        return (
            f"ERROR: {line_item} for {symbol} has no value recorded for {period}."
        )
    unit_suffix = f" {unit}" if unit else ""
    return f"{line_item} for {symbol} in {period}: {value}{unit_suffix} (source: {table})"


@tool("search_annual_report", args_schema=AnnualReportSearchInput)
def search_annual_report(
    symbol: str, query: str, year: str | None = None, n_results: int = 3
) -> str:
    """Search a company's annual report text for narrative/qualitative content.

    Use this for non-numeric questions such as management discussion,
    strategy, risks, or outlook. Do not use this for verified financial
    figures - use get_financial_data instead.

    Args:
        symbol: Company ticker/screener symbol.
        query: Natural-language question to search for.
        year: Optional 4-digit annual report year to filter to.
        n_results: Number of matching sections to retrieve.

    Returns:
        Concatenated parent-context text from the best-matching report
        sections, or an "ERROR: ..." string if the search failed or
        returned nothing.
    """
    where: dict[str, object] = {"company": symbol}
    if year is not None:
        where["year"] = int(year)

    try:
        store = ChromaStore.get_instance()
        results = store.query_children_with_parent_context(
            query_texts=[query], n_results=n_results, where=where
        )
    except Exception as exc:
        logger.exception(
            "Vector search failed for symbol=%s year=%s query=%r", symbol, year, query
        )
        return f"ERROR: annual report search failed for {symbol}: {exc}"

    if not results:
        return f"ERROR: no annual report content found for {symbol} matching the query."

    sections = []
    for item in results:
        text = item.get("text") if isinstance(item, dict) else None
        if text:
            sections.append(text)

    if not sections:
        return f"ERROR: no annual report content found for {symbol} matching the query."

    return "\n\n---\n\n".join(sections)