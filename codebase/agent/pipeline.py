"""Multi-agent pipeline orchestrator.

Wires together every stage built for the elaborative/reasoning system:

    1. Query Understanding   (query_understanding.understand_query)      - LLM
    2. Clarification Gate    (clarification.resolve_query)               - deterministic
    3. Cache check           (CacheMemory.build_structured_cache_key)    - deterministic
    4. Data Retrieval         (series_tools.fetch_financial_series)       - deterministic
       + Enrichment           (enrichment.enrich_series)                  - deterministic
    5. Context Retrieval     (context_retrieval.retrieve_context)        - LLM tool (conditional)
    6. Synthesis              (synthesis.synthesize_answer)               - LLM, no tools
    7. Follow-up Suggestor    (followup.suggest_follow_ups)                - deterministic
    8. Cache write            (CacheMemory.set_cached_response)           - deterministic

Stage 4 is called directly in code rather than via an LLM tool-calling
loop: Stage 2 already produces an exact symbol + line_items + periods
list, so there is nothing left for an LLM to decide at this point - adding
a tool-calling agent here would only add latency and a new place for a
malformed tool call, with no benefit. This is consistent with, not a
deviation from, the "no model judgment where a deterministic answer
exists" principle applied throughout this design.

CACHE KEY: uses CacheMemory.build_structured_cache_key (the new ADDITIVE
method - see codebase/agentmemory/cachememory.py). The key is built from
(symbol, line_items, periods, needs_qualitative_context, intent) - NOT
raw question text - so two differently-worded questions that resolve to
the same company/metrics/periods correctly share one cache entry. The
ORIGINAL build_cache_key method, and every other method on CacheMemory,
is completely untouched; this orchestrator is simply a new caller using
the new method, exactly like runner.py uses the old one.
"""

from __future__ import annotations

import logging

from codebase.agent.clarification import resolve_query
from codebase.agent.context_retrieval import retrieve_context
from codebase.agent.enrichment import EnrichedSeries, enrich_series
from codebase.agent.followup import suggest_follow_ups
from codebase.agent.query_understanding import understand_query
from codebase.agent.series_tools import fetch_financial_series
from codebase.agent.synthesis import synthesize_answer
from codebase.agentmemory import CacheMemory

logger = logging.getLogger(__name__)

# Single module-level CacheMemory instance, same pattern as runner.py's
# module-level _cache and agents.py's module-level agent singletons.
_cache = CacheMemory()


def _build_cache_key(symbol: str, line_items: list[str], periods: list[str],
                      needs_qualitative_context: bool, intent: str) -> tuple[str, dict]:
    """Build the structured, phrasing-independent cache key for this query.

    Sorting line_items/periods before hashing means two requests for the
    same metrics/periods in a different extraction ORDER still hit the
    same cache entry - the resolved SET is the identity, not the sequence
    in which Stage 1 happened to list them.
    """
    return _cache.build_structured_cache_key(
        company=symbol,
        extra_filters={
            "line_items": sorted(line_items),
            "periods": sorted(periods),
            "needs_qualitative_context": needs_qualitative_context,
            "intent": intent,
        },
    )


def answer_query(query: str) -> dict:
    """Run the full multi-agent pipeline for one user query.

    Returns
    -------
    dict
        {
          "answer": str,               # final synthesized answer, or a stop/clarification message
          "stopped": bool,             # True if the pipeline halted at the clarification gate
          "from_cache": bool,          # True if served from cache without invoking any agent
          "suggestions": list[str],    # follow-up question suggestions (empty if stopped)
          "symbol": str | None,
          "line_items": list[str],
          "periods": list[str],
        }
    """
    understanding = understand_query(query)
    resolved = resolve_query(understanding)

    if not resolved.can_proceed:
        # Per product decision (carried over from runner.py): a stop/
        # clarification message is deterministic and instant - never
        # cached, since caching it adds staleness risk for zero benefit.
        return {
            "answer": resolved.stop_message,
            "stopped": True,
            "from_cache": False,
            "suggestions": [],
            "symbol": None,
            "line_items": [],
            "periods": [],
        }

    cache_key, normalized_payload = _build_cache_key(
        resolved.symbol, resolved.line_items, resolved.periods,
        resolved.needs_qualitative_context, resolved.intent,
    )

    cached = _cache.get_cached_response(cache_key)
    if cached is not None:
        logger.info(
            "Cache HIT for symbol=%s line_items=%s periods=%s (hit_count=%s)",
            resolved.symbol, resolved.line_items, resolved.periods, cached.get("hit_count"),
        )
        return {
            "answer": cached["response"]["answer"],
            "stopped": False,
            "from_cache": True,
            "suggestions": cached["response"].get("suggestions", []),
            "symbol": resolved.symbol,
            "line_items": resolved.line_items,
            "periods": resolved.periods,
        }

    logger.info(
        "Cache MISS for symbol=%s line_items=%s periods=%s - running pipeline",
        resolved.symbol, resolved.line_items, resolved.periods,
    )

    # --- Stage 4: Data Retrieval + deterministic enrichment ---
    series_list: list[EnrichedSeries] = []
    fetch_errors: list[str] = []
    for line_item in resolved.line_items:
        fetch_result = fetch_financial_series(resolved.symbol, line_item, resolved.periods)
        if not fetch_result.get("ok"):
            fetch_errors.append(fetch_result.get("error", f"Unknown error fetching {line_item}"))
            continue
        enriched = enrich_series(fetch_result, resolved.periods)
        if enriched is not None:
            series_list.append(enriched)

    if not series_list:
        # Every requested line_item failed to resolve (e.g. all misspelled
        # or genuinely not tracked for this company). Distinct from "found
        # the company but found nothing" at the clarification gate - this
        # is a data-availability failure for the SPECIFIC metrics asked,
        # discovered only after Stage 4 tried. Not cached, same reasoning
        # as the clarification-gate stop above.
        error_summary = "; ".join(fetch_errors) if fetch_errors else "no data found"
        return {
            "answer": f"I couldn't retrieve the requested data for {resolved.symbol}: {error_summary}",
            "stopped": True,
            "from_cache": False,
            "suggestions": [],
            "symbol": resolved.symbol,
            "line_items": resolved.line_items,
            "periods": resolved.periods,
        }

    # --- Stage 5: Context Retrieval (conditional) ---
    context_snippets: list[dict] = []
    if resolved.needs_qualitative_context:
        context_result = retrieve_context(
            resolved.symbol, resolved.line_items, resolved.periods, query
        )
        context_snippets = context_result["snippets"]
        if context_result["errors"]:
            logger.info(
                "Context retrieval had non-fatal issues for symbol=%s: %s",
                resolved.symbol, context_result["errors"],
            )

    # --- Stage 6: Synthesis ---
    answer_text = synthesize_answer(
        query, resolved.symbol, series_list,
        context_snippets=context_snippets if context_snippets else None,
    )

    # --- Stage 7: Follow-up Suggestor (deterministic) ---
    suggestions = suggest_follow_ups(
        resolved.symbol, resolved.line_items, resolved.periods,
        qualitative_context_was_used=bool(context_snippets),
    )

    # --- Stage 8: Cache write ---
    # Only cache a real, non-empty answer - same guard as runner.py, so a
    # transient synthesis failure can't poison the cache for a query that
    # might succeed on retry.
    if answer_text:
        _cache.set_cached_response(
            cache_key=cache_key,
            normalized_payload=normalized_payload,
            original_question=query,
            response={"status": "success", "answer": answer_text, "suggestions": suggestions},
        )
        logger.info(
            "Cached new answer for symbol=%s line_items=%s periods=%s",
            resolved.symbol, resolved.line_items, resolved.periods,
        )

    return {
        "answer": answer_text,
        "stopped": False,
        "from_cache": False,
        "suggestions": suggestions,
        "symbol": resolved.symbol,
        "line_items": resolved.line_items,
        "periods": resolved.periods,
    }
