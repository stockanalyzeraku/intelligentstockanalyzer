"""Stage 6: Synthesis Agent.

Writes the final user-facing answer. Has NO TOOL ACCESS - this is the
structural guarantee that it cannot fetch or "remember" a number that
wasn't already verified by Stage 4 + enrichment.py. It is handed:
  - an EnrichedSeries per line_item (raw values + already-computed YoY/
    CAGR/direction - see enrichment.py), and
  - optional grounded text snippets from Stage 5 (only if
    needs_qualitative_context was true).

Per product decisions made earlier in this design:
  - Only synthesizes across line_items/periods the user actually asked
    about. Never introduces other metrics it notices moved unusually.
  - No peer/industry comparison.
  - No proactive uncertainty/caveat flagging beyond what the data itself
    already represents (e.g. a None value for "not disclosed" is stated
    plainly, not editorialized about).
  - Never computes a number itself - every number in its prompt context is
    already final; its job is narration and reasoning ABOUT given numbers,
    not arithmetic.
"""

from __future__ import annotations

import logging

from config import CONFIG
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI

from codebase.agent.enrichment import EnrichedSeries

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SYNTHESIS = """You write the final answer to a financial \
question about an Indian listed company, using ONLY the verified data \
given to you in this conversation. You have NO tools - you cannot look \
anything up. Every number you state MUST already appear in the data you \
were given.

Rules:
- State the requested figures clearly, with their unit and period.
- If multiple periods are given, describe the trend using the PRE-COMPUTED \
year-over-year changes, overall change, and CAGR you were given - do NOT \
calculate any new percentage or growth figure yourself. If a computed \
figure (e.g. CAGR) is given as "not available", say so plainly rather \
than estimating one.
- If a value swings from negative to positive (or vice versa) between \
periods, describe this in words (e.g. "swung from a loss to a profit") \
rather than leaning on a percentage figure, since percentages across a \
sign change can be misleading.
- If a period's value is marked as not disclosed/missing, state that \
plainly. Do not guess, interpolate, or estimate a missing value.
- ONLY discuss the line item(s) and period(s) you were explicitly given. \
Do NOT mention or speculate about other financial metrics, even if you \
suspect they are relevant, unless they were explicitly included in your \
given data.
- Do NOT compare this company to its industry, peers, or competitors. \
Discuss only this company's own figures.
- Do NOT add unprompted caveats, warnings, or "this looks unusual" \
commentary beyond plainly stating what the data shows. If something is \
missing, say it is missing - do not speculate about why.
- If qualitative context (annual report excerpts) is provided, you may use \
it to explain WHY a number changed, but only in direct service of \
answering what was asked - do not import the qualitative context as \
numeric data. Any number must still come from the verified data, never \
from the qualitative text.
- Keep the answer clear, well-reasoned, and grounded. Do not pad with \
generic filler.
"""

_model = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=CONFIG.MISTRAL_API_KEY,
)

synthesis_agent = create_agent(
    model=_model,
    tools=[],
    system_prompt=SYSTEM_PROMPT_SYNTHESIS,
)


def _format_series_for_prompt(series: EnrichedSeries) -> str:
    """Render an EnrichedSeries as plain text for the synthesis prompt.

    Deliberately verbose/explicit (e.g. spells out "not disclosed" and
    "not available" rather than leaving gaps) so the LLM has no ambiguity
    to fill in with a guess.
    """
    lines = [f"Line item: {series.line_item}", f"Unit: {series.unit or 'unknown'}"]
    for point in series.points:
        value_str = f"{point.value}" if point.value is not None else "not disclosed"
        yoy_str = (
            f" (YoY change vs previous period: {point.yoy_change_pct:+.2f}%)"
            if point.yoy_change_pct is not None
            else ""
        )
        lines.append(f"  {point.period}: {value_str}{yoy_str}")

    first_to_last = (
        f"{series.first_to_last_change_pct:+.2f}%"
        if series.first_to_last_change_pct is not None
        else "not available (insufficient data)"
    )
    cagr = f"{series.cagr_pct:+.2f}%" if series.cagr_pct is not None else "not available"
    lines.append(f"  Overall change (first to last available period): {first_to_last}")
    lines.append(f"  CAGR: {cagr}")
    lines.append(f"  Direction: {series.direction}")
    return "\n".join(lines)


def _format_context_snippets(snippets: list[dict]) -> str:
    if not snippets:
        return "(No qualitative annual-report context was found or requested for this question.)"
    parts = []
    for s in snippets:
        parts.append(f"[{s['line_item']} / {s['period']}]: {s['text']}")
    return "\n\n".join(parts)


def synthesize_answer(
    query: str,
    symbol: str,
    series_list: list[EnrichedSeries],
    context_snippets: list[dict] | None = None,
) -> str:
    """Run the Synthesis agent and return the final answer text.

    Parameters
    ----------
    query : str
        The user's original question, given as context for tone/focus -
        the agent should still only narrate the data it's given, not treat
        this as license to answer anything beyond that data.
    symbol : str
        Resolved company symbol, for the prompt's framing only.
    series_list : list[EnrichedSeries]
        One enriched series per line_item that was actually resolved and
        fetched in Stage 4. This is the ONLY source of numbers the agent
        may use.
    context_snippets : list[dict], optional
        Output of context_retrieval.retrieve_context()["snippets"], if
        Stage 5 ran. Empty/None if not applicable.

    Returns
    -------
    str
        The final answer text. On an unexpected agent failure, returns a
        plain data-only fallback string built directly from series_list
        (never silent failure, never a fabricated answer).
    """
    series_text = "\n\n".join(_format_series_for_prompt(s) for s in series_list)
    context_text = _format_context_snippets(context_snippets or [])

    prompt = (
        f"User's question: {query}\n\n"
        f"Company: {symbol}\n\n"
        f"Verified financial data (the ONLY numbers you may use):\n{series_text}\n\n"
        f"Qualitative context (annual report excerpts, optional):\n{context_text}\n\n"
        f"Write the final answer now."
    )

    try:
        result = synthesis_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        content_blocks = result["messages"][-1].content_blocks
        return _flatten_content_blocks(content_blocks)
    except Exception:  # noqa: BLE001 - this stage must never crash the pipeline
        logger.exception("Synthesis agent failed for query=%r symbol=%s", query, symbol)
        return _fallback_answer(series_list)


def _flatten_content_blocks(content_blocks) -> str:
    """Same flattening logic as runner.py - kept local to avoid a cross-
    module dependency for one small helper. See runner.py for the
    documented rationale (only "text"-type blocks contribute to the answer).
    """
    if isinstance(content_blocks, str):
        return content_blocks
    if not isinstance(content_blocks, list):
        return str(content_blocks)
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts).strip()


def _fallback_answer(series_list: list[EnrichedSeries]) -> str:
    """Plain, data-only answer with no LLM involvement, used only if the
    synthesis agent call itself fails outright. Less fluent than a real
    synthesis, but never wrong and never silent.
    """
    parts = []
    for series in series_list:
        values = ", ".join(
            f"{p.period}: {p.value if p.value is not None else 'not disclosed'} {series.unit or ''}".strip()
            for p in series.points
        )
        parts.append(f"{series.line_item}: {values}")
    return "I couldn't generate a full narrative answer, but here is the verified data:\n" + "\n".join(parts)
