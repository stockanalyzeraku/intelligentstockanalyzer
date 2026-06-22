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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_QUERY_UNDERSTANDING = """You extract structured information \
from a user's question about an Indian listed company's financial \
statements. You do NOT answer the question. You ONLY extract fields.

Rules:
- symbol: resolve the company name to its known ticker/screener symbol if \
you recognize it (e.g. "Kalyan Jewellers" -> "KALYANKJIL"). If you do not \
clearly recognize the company, leave symbol as null and set \
ambiguity_reason explaining that the company could not be identified. \
Never invent a symbol you are not confident about.
- line_items: list ONLY the specific financial metrics the user named or \
clearly implied (e.g. "sales", "revenue" -> "Sales"; "profit" -> "Net \
Profit"; "eps" -> "EPS in Rs"). Do NOT add related metrics the user did \
not ask about.
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


def understand_query(query: str) -> QueryUnderstanding:
    """Run the Query Understanding agent on a raw user query.

    Returns
    -------
    QueryUnderstanding
        Always a valid instance. If the underlying agent call fails or
        returns something that doesn't validate, returns a safe fallback
        instance with ambiguity_reason set, rather than raising - so the
        pipeline can always proceed to its clarification-gate logic
        without a try/except at every call site.
    """
    try:
        result = query_understanding_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        structured = result.get("structured_response")
        if isinstance(structured, QueryUnderstanding):
            return structured
        # Some provider strategies may return a dict instead of the
        # validated model instance depending on LangChain version/provider
        # path - handle that defensively rather than assuming.
        if isinstance(structured, dict):
            return QueryUnderstanding(**structured)
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
