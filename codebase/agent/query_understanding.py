"""Stage 1: Query Understanding agent.

Single-purpose LLM call: turn a raw user question into a validated
QueryUnderstanding object (see schemas.py). This agent has NO tools - its
only job is structured extraction, not answering anything. Code
immediately downstream (pipeline.py) validates/defaults the fields and
decides what happens next; the LLM's role stops at extraction.

Built via create_agent(..., response_format=QueryUnderstanding), which
(per LangChain's structured-output docs) returns the validated Pydantic
instance in result["structured_response"].

NOTE: structured-output reliability via response_format varies by model
provider - some providers support it natively (most reliable), others
fall back to a tool-calling-based strategy. This has not yet been
exercised against the live Mistral model configured in agents.py; verify
this works as expected once real API access is available, and watch for
LangChain issues around `response_format` + provider quirks (e.g. some
providers don't support tools + structured output simultaneously without
specific provider_strategy configuration).
"""

from __future__ import annotations

import json
import logging

from config import CONFIG
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI

from codebase.agent.schemas import QueryUnderstanding
from codebase.financials.aliases import init_aliases_schema

logger = logging.getLogger(__name__)

# Ensure the company_aliases table exists before _check_known_aliases /
# _maybe_save_alias ever run. This module can be imported on its own
# (e.g. by a test, or a future entry point other than pipeline.py) without
# pipeline.py's own init_aliases_schema() call having run first - calling
# it here too makes this module self-sufficient regardless of import
# order. init_aliases_schema() is idempotent (CREATE TABLE IF NOT EXISTS),
# so calling it from two places is safe, not wasteful in any meaningful way.
init_aliases_schema()

SYSTEM_PROMPT_QUERY_UNDERSTANDING = """You extract structured information \
from a user's question about an Indian listed company's financial \
statements. You do NOT answer the question. You ONLY extract fields.

Rules:
- symbol: resolve the company name to its known ticker/screener symbol if \
you recognize it (e.g. "Kalyan Jewellers" -> "KALYANKJIL"). If you do not \
clearly recognize the company, leave symbol as null and set \
ambiguity_reason explaining that the company could not be identified. \
Never invent a symbol you are not confident about.
- confident: set this True ONLY when you are highly sure of the symbol \
resolution (well-known company name, exact ticker, or a name/alias you \
were explicitly told maps to this symbol). Set False if you had to guess \
or infer from a partial/unusual phrasing, even if you did still produce a \
symbol. When in doubt, prefer False - this only affects whether the \
resolution gets remembered for next time, not whether you can answer now.
- line_items: list ONLY the specific financial metrics the user named or \
clearly implied (e.g. "sales", "revenue" -> "Sales"; "profit" -> "Net \
Profit"; "eps" -> "EPS in Rs"; "operating margin"/"opm" -> "OPM %"). Do \
NOT add related metrics the user did not ask about.
- derived_metrics: list ONLY computed ratios the user asked about that are \
NOT directly published figures: "Net Profit Margin", "ROE" (or "return on \
equity"), "Debt to Equity", "Free Cash Flow Margin". These are calculated, \
not looked up - keep them separate from line_items. Operating Margin/OPM \
is NOT a derived metric (it is published directly) - put it in line_items \
as "OPM %" instead.
- raw_years: list every 4-digit year explicitly mentioned. Leave empty if \
no year was mentioned at all - do not guess a year.
- comparison_requested: true only if the user explicitly asked to compare, \
see a trend, or see growth across periods.
- single_year_only: true only if the user explicitly said they want just \
one year with no additional context (e.g. "just 2023, nothing else").
- needs_qualitative_context: true if the question asks "why", for a \
reason, strategy, outlook, risk, or management commentary - not for a \
plain number lookup.
- intent: "financial" if a specific number is being asked about (even \
alongside qualitative context), "general" if it is purely narrative with \
no number requested.

Never answer the question itself. Never call any tool. Only return the \
structured fields.

If prior conversation context is given above the question, use it ONLY to
fill in a company/period/line_item the new question doesn't specify
itself. A company or period named explicitly in the new question always
overrides prior context - never let prior context override something the
user just said.
"""

_model = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=CONFIG.MISTRAL_API_KEY,
)

query_understanding_agent = create_agent(
    model=_model,
    tools=[],
    system_prompt=SYSTEM_PROMPT_QUERY_UNDERSTANDING,
    response_format=QueryUnderstanding,
)


def _check_known_aliases(query: str) -> str | None:
    """Best-effort, deterministic check: does the query contain a phrase
    we've already learned maps to a symbol? Substring match against the
    lowercased query, same simple approach as classify.py's existing
    COMPANY_LOOKUP dict - intentionally simple, not fuzzy, so it cannot
    silently misfire on an unrelated phrase.

    This does NOT replace the LLM call - it is only used to pre-fill a
    hint. The LLM still runs and still makes the final call, since the
    query may need other fields extracted (line_items, years, etc.)
    regardless of whether the company was already known. Returns None if
    no learned alias matches.
    """
    lowered = query.lower()
    try:
        from codebase.financials.aliases import list_aliases
        for alias in list_aliases():
            if alias["alias_text"] in lowered:
                return alias["screener_symbol"]
    except Exception:  # noqa: BLE001 - alias lookup is a convenience, never fatal
        logger.exception("Alias lookup failed for query=%r - continuing without it", query)
    return None


def understand_query(query: str, prior_state=None) -> QueryUnderstanding:
    """Run the Query Understanding agent on a raw user query.

    Parameters
    ----------
    query : str
        The raw user question.
    prior_state : ConversationState, optional
        Short-term state from earlier in this conversation (see
        conversation_state.py). If given, its rendered context is
        prepended to the prompt so an underspecified follow-up question
        (e.g. "what about net profit?") can fall back to the previously
        resolved company/period. A new company/period named explicitly in
        `query` always takes priority - this is instructed in the system
        prompt, not enforced here.

    Returns
    -------
    QueryUnderstanding
        Always a valid instance. If the underlying agent call fails or
        returns something that doesn't validate, returns a safe fallback
        instance with ambiguity_reason set, rather than raising - so the
        pipeline can always proceed to its clarification-gate logic
        without a try/except at every call site.

    Side effect: if the LLM resolves a company with confident=True AND
    company_name_as_given is set, the exact phrase the user typed is
    auto-saved as a new alias (codebase.financials.aliases.save_alias) for
    next time. This never overwrites an existing alias and never raises -
    see save_alias's own docstring for the full write policy.
    """
    known_alias_symbol = _check_known_aliases(query)

    prompt_content = query
    if prior_state is not None:
        prompt_content = f"{prior_state.as_prompt_context()}\n\nNew question: {query}"
    if known_alias_symbol is not None:
        prompt_content = (
            f"(Note: a prior conversation/usage already established that a "
            f"company name in this question refers to symbol "
            f"'{known_alias_symbol}' - use this if it matches what the "
            f"question is asking about.)\n\n{prompt_content}"
        )

    try:
        result = query_understanding_agent.invoke(
            {"messages": [{"role": "user", "content": prompt_content}]}
        )
        structured = result.get("structured_response")
        if isinstance(structured, QueryUnderstanding):
            _maybe_save_alias(structured, query)
            return structured
        # Some provider strategies may return a dict instead of the
        # validated model instance depending on LangChain version/provider
        # path - handle that defensively rather than assuming.
        if isinstance(structured, dict):
            parsed = QueryUnderstanding(**structured)
            _maybe_save_alias(parsed, query)
            return parsed
        logger.warning(
            "Query understanding returned unexpected structured_response type: %r",
            type(structured),
        )
    except Exception:  # noqa: BLE001 - this stage must never crash the pipeline
        logger.exception("Query understanding agent failed for query=%r", query)

    return QueryUnderstanding(
        ambiguity_reason=(
            "I had trouble understanding that question. Could you rephrase it, "
            "mentioning the company name explicitly?"
        )
    )


def _maybe_save_alias(structured: QueryUnderstanding, original_query: str) -> None:
    """Auto-save an alias ONLY when the model self-reported a confident
    resolution and gave us the exact phrase it resolved from. Best-effort:
    any failure here is logged and swallowed, never raised, since alias
    learning is a convenience and must never break query answering.
    """
    if not (structured.confident and structured.symbol and structured.company_name_as_given):
        return
    try:
        from codebase.financials.aliases import save_alias

        saved = save_alias(
            alias_text=structured.company_name_as_given,
            screener_symbol=structured.symbol,
            source="llm_confident",
            source_query=original_query,
        )
        if saved:
            logger.info(
                "Learned new alias: %r -> %s (from query: %r)",
                structured.company_name_as_given, structured.symbol, original_query,
            )
    except Exception:  # noqa: BLE001 - never let alias learning break the pipeline
        logger.exception("Failed to save alias for symbol=%s", structured.symbol)
