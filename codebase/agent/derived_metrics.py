"""Derived/ratio financial metrics, computed in code - NEVER by an LLM.

Same philosophy as enrichment.py: every number here is plain arithmetic
on values already fetched via fetch_financial_series(). The Synthesis
agent (Stage 6) only ever narrates what this module hands it; it never
computes a ratio itself.

Supported metrics (the agreed initial set):
    - Net Profit Margin   = Net Profit / Sales
    - Operating Margin    = NOT computed here - "OPM %" already exists as
                             a directly-published line item in
                             statement_profit_loss (screener.in publishes
                             it directly). Callers should fetch "OPM %" via
                             the normal series_tools path, not this module.
    - ROE (Return on Equity) = Net Profit / average(opening equity, closing equity)
                             where equity = Equity Capital + Reserves.
                             Uses AVERAGE equity (opening + closing / 2),
                             per standard convention and screener.in's own
                             methodology (which also averages capital
                             employed for ROE/ROCE) - NOT a simplified
                             single-period figure. This requires fetching
                             one extra prior-period balance sheet row
                             beyond what was otherwise requested.
    - Debt-to-Equity       = Borrowings / (Equity Capital + Reserves), using
                             ONLY the period's own closing balance (no
                             averaging - D/E is a point-in-time solvency
                             snapshot by standard convention, unlike ROE
                             which is a period-return metric).
    - Free Cash Flow Margin = Free Cash Flow / Sales

IMPORTANT - labeling requirement (product decision): every value produced
by this module MUST be presented as a computed/approximate figure, never
implied to be a number sourced directly from screener.in. Each DerivedMetric
below carries is_approximation=True and a human-readable label
(e.g. "ROE (computed, average equity)") for exactly this reason - the
Synthesis prompt is instructed to always show this label, never just the
bare metric name.
"""

from __future__ import annotations

from dataclasses import dataclass

from codebase.agent.series_tools import fetch_financial_series

# Maps a user-facing derived metric name to its internal key, so Stage 1's
# extraction ("ROE", "return on equity", "debt to equity") can be matched
# loosely while computation stays keyed on a fixed internal name.
DERIVED_METRIC_NAMES: dict[str, str] = {
    "net profit margin": "net_profit_margin",
    "npm": "net_profit_margin",
    "roe": "roe",
    "return on equity": "roe",
    "debt to equity": "debt_to_equity",
    "debt-to-equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "free cash flow margin": "fcf_margin",
    "fcf margin": "fcf_margin",
}

# Line items each derived metric needs fetched from each statement table.
# ROE additionally needs the PRIOR period's balance sheet row - handled
# specially in compute_derived_metric below, not listed here.
_METRIC_REQUIRED_LINE_ITEMS: dict[str, list[str]] = {
    "net_profit_margin": ["Net Profit", "Sales"],
    "roe": ["Net Profit", "Equity Capital", "Reserves"],
    "debt_to_equity": ["Borrowings", "Equity Capital", "Reserves"],
    "fcf_margin": ["Free Cash Flow", "Sales"],
}

DERIVED_METRIC_LABELS: dict[str, str] = {
    "net_profit_margin": "Net Profit Margin (computed: Net Profit / Sales)",
    "roe": "ROE (computed, average equity: Net Profit / avg of opening+closing Equity)",
    "debt_to_equity": "Debt-to-Equity (computed: Borrowings / Equity, closing balance)",
    "fcf_margin": "Free Cash Flow Margin (computed: Free Cash Flow / Sales)",
}


@dataclass
class DerivedMetricPoint:
    period: str
    value: float | None  # as a ratio (e.g. 0.18), NOT pre-multiplied by 100
    note: str | None = None  # set when the value couldn't be computed for this period


@dataclass
class DerivedMetric:
    metric_key: str
    label: str
    is_approximation: bool
    points: list[DerivedMetricPoint]
    formula_note: str


def resolve_derived_metric_name(name: str) -> str | None:
    """Match a free-text metric name (as extracted by Stage 1) to an
    internal metric_key, or None if it isn't a recognized derived metric.
    Case-insensitive, exact-phrase match against DERIVED_METRIC_NAMES.
    """
    return DERIVED_METRIC_NAMES.get(name.strip().lower())


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _prior_period(period_label: str) -> str | None:
    """'Mar 2023' -> 'Mar 2022'. Returns None if unparsable."""
    parts = period_label.strip().split()
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return f"{parts[0]} {int(parts[1]) - 1}"


def compute_derived_metric(metric_key: str, symbol: str, periods: list[str]) -> DerivedMetric | None:
    """Compute a derived metric across the given periods.

    Fetches whatever raw line items it needs via fetch_financial_series
    (the same verified data source Stage 4 uses for direct line items),
    then computes the ratio in plain Python. Returns None if metric_key
    isn't recognized.

    A period with missing underlying data produces a DerivedMetricPoint
    with value=None and a `note` explaining why, rather than silently
    omitting that period or guessing a value.
    """
    if metric_key not in _METRIC_REQUIRED_LINE_ITEMS:
        return None

    if metric_key == "roe":
        return _compute_roe(symbol, periods)
    if metric_key == "net_profit_margin":
        return _compute_simple_ratio(
            symbol, periods, numerator_item="Net Profit", denominator_item="Sales",
            metric_key="net_profit_margin",
        )
    if metric_key == "debt_to_equity":
        return _compute_debt_to_equity(symbol, periods)
    if metric_key == "fcf_margin":
        return _compute_simple_ratio(
            symbol, periods, numerator_item="Free Cash Flow", denominator_item="Sales",
            metric_key="fcf_margin",
        )
    return None  # pragma: no cover - unreachable given the guard above


def _compute_simple_ratio(
    symbol: str, periods: list[str], numerator_item: str, denominator_item: str, metric_key: str
) -> DerivedMetric:
    num_result = fetch_financial_series(symbol, numerator_item, periods)
    den_result = fetch_financial_series(symbol, denominator_item, periods)

    points = []
    for period in periods:
        num_value = num_result.get("values", {}).get(period) if num_result.get("ok") else None
        den_value = den_result.get("values", {}).get(period) if den_result.get("ok") else None
        ratio = _safe_divide(num_value, den_value)
        note = None
        if ratio is None:
            note = f"Could not compute - {numerator_item} or {denominator_item} not disclosed for {period}."
        points.append(DerivedMetricPoint(period=period, value=ratio, note=note))

    return DerivedMetric(
        metric_key=metric_key,
        label=DERIVED_METRIC_LABELS[metric_key],
        is_approximation=True,
        points=points,
        formula_note=f"{numerator_item} / {denominator_item}, per period.",
    )


def _compute_debt_to_equity(symbol: str, periods: list[str]) -> DerivedMetric:
    borrowings_result = fetch_financial_series(symbol, "Borrowings", periods)
    equity_capital_result = fetch_financial_series(symbol, "Equity Capital", periods)
    reserves_result = fetch_financial_series(symbol, "Reserves", periods)

    points = []
    for period in periods:
        borrowings = borrowings_result.get("values", {}).get(period) if borrowings_result.get("ok") else None
        equity_capital = equity_capital_result.get("values", {}).get(period) if equity_capital_result.get("ok") else None
        reserves = reserves_result.get("values", {}).get(period) if reserves_result.get("ok") else None

        equity = None
        if equity_capital is not None and reserves is not None:
            equity = equity_capital + reserves

        ratio = _safe_divide(borrowings, equity)
        note = None
        if ratio is None:
            note = f"Could not compute - Borrowings or Equity (Equity Capital + Reserves) not disclosed for {period}."
        points.append(DerivedMetricPoint(period=period, value=ratio, note=note))

    return DerivedMetric(
        metric_key="debt_to_equity",
        label=DERIVED_METRIC_LABELS["debt_to_equity"],
        is_approximation=True,
        points=points,
        formula_note="Borrowings / (Equity Capital + Reserves), closing balance, per period.",
    )


def _compute_roe(symbol: str, periods: list[str]) -> DerivedMetric:
    """ROE needs each period's OWN equity plus the PRIOR period's equity
    (for averaging) - so the actual fetch window is periods + one extra
    year before the earliest requested period.
    """
    prior_periods = [_prior_period(p) for p in periods]
    all_periods_needed = sorted({p for p in periods + prior_periods if p is not None})

    net_profit_result = fetch_financial_series(symbol, "Net Profit", periods)
    equity_capital_result = fetch_financial_series(symbol, "Equity Capital", all_periods_needed)
    reserves_result = fetch_financial_series(symbol, "Reserves", all_periods_needed)

    def _equity_at(period: str | None) -> float | None:
        if period is None:
            return None
        ec = equity_capital_result.get("values", {}).get(period) if equity_capital_result.get("ok") else None
        rs = reserves_result.get("values", {}).get(period) if reserves_result.get("ok") else None
        if ec is None or rs is None:
            return None
        return ec + rs

    points = []
    for period in periods:
        net_profit = net_profit_result.get("values", {}).get(period) if net_profit_result.get("ok") else None
        closing_equity = _equity_at(period)
        opening_equity = _equity_at(_prior_period(period))

        avg_equity = None
        note = None
        if closing_equity is not None and opening_equity is not None:
            avg_equity = (closing_equity + opening_equity) / 2
        elif closing_equity is not None and opening_equity is None:
            # Prior-period balance sheet not available (e.g. earliest
            # ingested year) - fall back to closing equity alone rather
            # than failing the whole computation, but say so explicitly.
            avg_equity = closing_equity
            note = (
                f"Prior-period balance sheet unavailable for {period} - "
                f"using closing equity only, not the average."
            )

        ratio = _safe_divide(net_profit, avg_equity)
        if ratio is None and note is None:
            note = f"Could not compute - Net Profit or Equity not disclosed for {period}."

        points.append(DerivedMetricPoint(period=period, value=ratio, note=note))

    return DerivedMetric(
        metric_key="roe",
        label=DERIVED_METRIC_LABELS["roe"],
        is_approximation=True,
        points=points,
        formula_note=(
            "Net Profit / average(opening Equity, closing Equity), where "
            "Equity = Equity Capital + Reserves."
        ),
    )
