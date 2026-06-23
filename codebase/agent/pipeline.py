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

MEMORY (added on top of the above, all additive):
  - Short-term state: ConversationState (conversation_state.py) is
    caller-owned, in-memory only, lives exactly as long as one
    conversation. Pass the "state" key from a previous answer_query()
    result back in as `prior_state` to let an underspecified follow-up
    ("what about net profit?") resolve against the previous turn's
    company/period. Never persisted, never shared between sessions.
  - Long-term preferences: UserPreferences (agentmemory/preferences.py)
    is a new, additive module - does not touch cachememory.py/
    cachestructure.py/dbstructure.py/workingmemory.py. Single hardcoded
    user_id for now (DEFAULT_USER_ID) since there are no real user
    accounts yet. Preferences only ever set DEFAULTS (trailing years,
    whether qualitative context is always pulled in) - they never
    override an explicit instruction in the query itself.
  - Long-term aliases: codebase/financials/aliases.py (lives in
    financials/, not agentmemory/, since it's entity-resolution data, not
    conversational memory) learns alias_text -> screener_symbol mappings
    when Stage 1 self-reports a confident resolution. See
    query_understanding.py's _maybe_save_alias for the write policy.
"""

from __future__ import annotations

import logging

from codebase.agent.clarification import DEFAULT_TRAILING_YEARS, resolve_query
from codebase.agent.context_retrieval import retrieve_context
from codebase.agent.conversation_state import ConversationState
from codebase.agent.derived_metrics import compute_derived_metric, resolve_derived_metric_name
from codebase.agent.enrichment import EnrichedSeries, enrich_series
from codebase.agent.followup import suggest_follow_ups
from codebase.agent.query_understanding import understand_query
from codebase.agent.series_tools import fetch_financial_series
from codebase.agent.synthesis import synthesize_answer
from codebase.agentmemory import CacheMemory
from codebase.agentmemory.preferences import DEFAULT_USER_ID, PreferenceKeys, UserPreferences
from codebase.financials.aliases import init_aliases_schema

logger = logging.getLogger(__name__)

# Single module-level singletons, same pattern as runner.py's module-level
# _cache and agents.py's module-level agent singletons.
_cache = CacheMemory()
_preferences = UserPreferences()

# Ensure the company_aliases table exists before any query is answered.
# query_understanding.py reads/writes this table via the standalone
# functions in financials/aliases.py (resolve_alias, save_alias,
# list_aliases) rather than a class instance, so unlike CacheMemory()/
# UserPreferences() above there's no constructor that creates the table
# as a side effect - it must be created explicitly, once, here.
init_aliases_schema()


def _build_cache_key(symbol: str, line_items: list[str], derived_metric_keys: list[str],
                      periods: list[str], needs_qualitative_context: bool, intent: str) -> tuple[str, dict]:
    """Build the structured, phrasing-independent cache key for this query.

    Sorting line_items/derived_metric_keys/periods before hashing means two
    requests for the same metrics/periods in a different extraction ORDER
    still hit the same cache entry - the resolved SET is the identity, not
    the sequence in which Stage 1 happened to list them.
    """
    return _cache.build_structured_cache_key(
        company=symbol,
        extra_filters={
            "line_items": sorted(line_items),
            "derived_metrics": sorted(derived_metric_keys),
            "periods": sorted(periods),
            "needs_qualitative_context": needs_qualitative_context,
            "intent": intent,
        },
    )


def answer_query(query: str, prior_state: ConversationState | None = None, user_id: str = DEFAULT_USER_ID) -> dict:
    """Run the full multi-agent pipeline for one user query.

    Parameters
    ----------
    query : str
        The raw user question.
    prior_state : ConversationState, optional
        Short-term state from earlier in this conversation (see
        conversation_state.py). The CALLER owns this - pass back in
        whatever was returned as "state" from the previous call to keep a
        conversation going, or omit/pass None to start fresh. This
        pipeline holds no hidden state of its own between calls.
    user_id : str
        Defaults to the single-user constant (see
        codebase.agentmemory.preferences.DEFAULT_USER_ID) - this system
        does not yet have real multi-user accounts. Preferences are read
        for this user_id and only ever affect DEFAULTS (trailing years,
        whether qualitative context is always included) - they never
        override something explicit in the query itself.

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
          "state": ConversationState,  # pass this back in as prior_state on the NEXT call
        }
    """
    trailing_years = _preferences.get_preference(
        PreferenceKeys.TRAILING_YEARS, default=DEFAULT_TRAILING_YEARS, user_id=user_id
    )
    always_include_qualitative = _preferences.get_preference(
        PreferenceKeys.ALWAYS_INCLUDE_QUALITATIVE, default=False, user_id=user_id
    )

    understanding = understand_query(query, prior_state=prior_state)
    resolved = resolve_query(understanding, trailing_years=trailing_years)

    if always_include_qualitative and resolved.can_proceed:
        resolved.needs_qualitative_context = True

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
            "state": prior_state if prior_state is not None else ConversationState(),
        }

    # Resolve free-text derived metric names (from Stage 1) to internal
    # keys up front, since both the cache key and Stage 4 computation need
    # the canonical key, not the user's original phrasing.
    resolved_derived_keys: list[str] = []
    unresolved_derived_names: list[str] = []
    for name in resolved.derived_metrics:
        key = resolve_derived_metric_name(name)
        if key is not None and key not in resolved_derived_keys:
            resolved_derived_keys.append(key)
        elif key is None:
            unresolved_derived_names.append(name)

    if unresolved_derived_names:
        logger.info(
            "Unrecognized derived metric name(s) %s for symbol=%s - ignoring, not treated as a stop condition",
            unresolved_derived_names, resolved.symbol,
        )

    cache_key, normalized_payload = _build_cache_key(
        resolved.symbol, resolved.line_items, resolved_derived_keys, resolved.periods,
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
            "state": ConversationState.from_resolved_query(resolved),
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

    derived_metric_results = []
    for metric_key in resolved_derived_keys:
        derived = compute_derived_metric(metric_key, resolved.symbol, resolved.periods)
        if derived is not None:
            derived_metric_results.append(derived)

    if not series_list and not derived_metric_results:
        # Every requested line_item/derived_metric failed to resolve (e.g.
        # all misspelled or genuinely not tracked for this company).
        # Distinct from "found the company but found nothing" at the
        # clarification gate - this is a data-availability failure for the
        # SPECIFIC metrics asked, discovered only after Stage 4 tried. Not
        # cached, same reasoning as the clarification-gate stop above.
        error_summary = "; ".join(fetch_errors) if fetch_errors else "no data found"
        return {
            "answer": f"I couldn't retrieve the requested data for {resolved.symbol}: {error_summary}",
            "stopped": True,
            "from_cache": False,
            "suggestions": [],
            "symbol": resolved.symbol,
            "line_items": resolved.line_items,
            "periods": resolved.periods,
            # The company itself WAS validly resolved here (only the
            # specific line_item/derived_metric failed) - keep symbol and
            # periods in state so a follow-up question doesn't lose that
            # context, but don't carry forward the line_items that just
            # failed to resolve.
            "state": ConversationState(
                symbol=resolved.symbol, periods=resolved.periods,
                needs_qualitative_context=resolved.needs_qualitative_context,
            ),
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
        derived_metrics=derived_metric_results if derived_metric_results else None,
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
            "Cached new answer for symbol=%s line_items=%s derived_metrics=%s periods=%s",
            resolved.symbol, resolved.line_items, resolved_derived_keys, resolved.periods,
        )

    return {
        "answer": answer_text,
        "stopped": False,
        "from_cache": False,
        "suggestions": suggestions,
        "symbol": resolved.symbol,
        "line_items": resolved.line_items,
        "periods": resolved.periods,
        "state": ConversationState.from_resolved_query(resolved),
    }
