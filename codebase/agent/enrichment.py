"""Deterministic post-processing of a fetched financial series.

Per product decision: the Synthesis agent (Stage 6) NEVER computes a
number itself - not even simple arithmetic like a YoY percentage. Every
number it narrates, including deltas and growth rates, must already exist
in the data it's handed. This module is where that arithmetic happens, in
plain Python, immediately after Stage 4's retrieval and before Stage 6
ever sees the result.

This keeps the "no number from the model's head" guarantee airtight: if
the LLM only ever narrates pre-computed values, there's no path for it to
silently miscalculate or invent a percentage.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PeriodValue:
    """One (period, value) point in a series, after enrichment."""

    period: str
    value: float | None
    yoy_change_pct: float | None = None  # % change vs the previous period in this series, if both exist


@dataclass
class EnrichedSeries:
    """A fully computed series for one line_item, ready to hand to Stage 6.

    All fields here are either raw retrieved data or simple, deterministic
    arithmetic on that data - nothing in this object was decided by an LLM.
    """

    symbol: str
    line_item: str
    unit: str | None
    table: str
    points: list[PeriodValue] = field(default_factory=list)
    first_to_last_change_pct: float | None = None
    cagr_pct: float | None = None
    direction: str = "unknown"  # "up" | "down" | "flat" | "unknown" (unknown = insufficient data)


def _pct_change(old: float | None, new: float | None) -> float | None:
    """(new - old) / abs(old) * 100, or None if either is missing or old is 0."""
    if old is None or new is None or old == 0:
        return None
    return ((new - old) / abs(old)) * 100.0


def _cagr_pct(first: float | None, last: float | None, num_periods: int) -> float | None:
    """Compound annual growth rate as a percentage.

    num_periods is the number of YEAR STEPS between first and last (e.g. 3
    points spanning Mar 2021 -> Mar 2023 is 2 steps), not the count of points.
    Returns None if inputs are missing, non-positive (CAGR is undefined for
    a sign change or zero base), or num_periods <= 0.
    """
    if first is None or last is None or num_periods <= 0:
        return None
    if first <= 0 or last <= 0:
        # CAGR is mathematically undefined/misleading across a sign change
        # (e.g. loss -> profit). Leave it unset rather than compute a
        # number that could be misread by the synthesis step.
        return None
    return ((last / first) ** (1.0 / num_periods) - 1.0) * 100.0


def _infer_year_step_count(periods_in_order: list[str]) -> int:
    """Best-effort count of year-steps spanned by an ordered period list.

    Periods are expected as "Mon YYYY" labels (e.g. "Mar 2021"). Falls back
    to (count - 1) if year parsing fails for any label, which is correct
    for the common case of consecutive annual periods anyway.
    """
    years = []
    for p in periods_in_order:
        parts = p.strip().split()
        if len(parts) == 2 and parts[1].isdigit():
            years.append(int(parts[1]))
    if len(years) == len(periods_in_order) and len(years) >= 2:
        return years[-1] - years[0]
    return max(len(periods_in_order) - 1, 0)


def enrich_series(fetch_result: dict, periods_in_order: list[str]) -> EnrichedSeries | None:
    """Turn a fetch_financial_series() result into an EnrichedSeries.

    Parameters
    ----------
    fetch_result : dict
        The dict returned by codebase.agent.series_tools.fetch_financial_series.
        Must have "ok": True - callers should check this before enriching.
    periods_in_order : list[str]
        The periods in chronological order (oldest first), e.g.
        ["Mar 2021", "Mar 2022", "Mar 2023"]. Determines point order and
        which two points anchor first_to_last_change_pct/cagr_pct.

    Returns
    -------
    EnrichedSeries, or None if fetch_result wasn't a successful ("ok": True) result.
    """
    if not fetch_result.get("ok"):
        return None

    values = fetch_result["values"]
    points: list[PeriodValue] = []
    previous_value: float | None = None

    for period in periods_in_order:
        value = values.get(period)
        yoy = _pct_change(previous_value, value) if previous_value is not None else None
        points.append(PeriodValue(period=period, value=value, yoy_change_pct=yoy))
        if value is not None:
            previous_value = value

    # Anchor first-to-last comparison on the first and last points that
    # actually HAVE a value, not just the first/last requested period
    # (a period with no disclosed value shouldn't silently anchor a
    # comparison to None).
    valued_points = [p for p in points if p.value is not None]
    first_to_last_pct = None
    cagr = None
    direction = "unknown"

    if len(valued_points) >= 2:
        first_point, last_point = valued_points[0], valued_points[-1]
        first_to_last_pct = _pct_change(first_point.value, last_point.value)
        step_count = _infer_year_step_count([first_point.period, last_point.period])
        cagr = _cagr_pct(first_point.value, last_point.value, step_count)

        if first_to_last_pct is not None:
            if first_to_last_pct > 0.5:
                direction = "up"
            elif first_to_last_pct < -0.5:
                direction = "down"
            else:
                direction = "flat"

    return EnrichedSeries(
        symbol=fetch_result["symbol"],
        line_item=fetch_result["line_item"],
        unit=fetch_result["unit"],
        table=fetch_result["table"],
        points=points,
        first_to_last_change_pct=first_to_last_pct,
        cagr_pct=cagr,
        direction=direction,
    )
