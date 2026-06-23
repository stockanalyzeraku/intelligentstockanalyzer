"""Mocked eval suite for the multi-agent pipeline.

Run every time, free, fast, no API key needed:

    python -m unittest codebase.agent.evals.test_pipeline_mocked -v

or with pytest:

    pytest codebase/agent/evals/test_pipeline_mocked.py -v

WHAT THIS TESTS (real, not mocked):
    - clarification.py's trailing-3-year default, single_year_only,
      explicit multi-year, and unresolvable-company handling
    - series_tools.fetch_financial_series against the real financials DB
      (auto-discovering the correct statement table, real units)
    - enrichment.py's YoY/CAGR/direction math (hand-verified)
    - derived_metrics.py's ROE/Net Profit Margin/Debt-to-Equity/FCF Margin
      computation against real data (hand-verified)
    - followup.py's deterministic, data-backed suggestions
    - cachememory.py's NEW build_structured_cache_key (phrasing-independent
      caching) - the ORIGINAL build_cache_key and every other method is
      untouched and not re-tested here (see test_cachememory_regression.py)
    - pipeline.py's full orchestration: cache hit/miss, Stage 5 gating,
      derived-metric-only queries, stop conditions

WHAT THIS DOES NOT TEST:
    - Whether a REAL Mistral model actually extracts the right structured
      fields, or actually obeys the synthesis system prompt's rules (no
      model arithmetic, approximation labeling, no cross-metric
      speculation). The Query Understanding and Synthesis agents are
      MOCKED here with canned/passthrough responses. See eval_real_api.py
      for a real-model spot-check, run separately and occasionally.

PREREQUISITE: a financials.db with RELIANCE ingested must exist at the
default path (codebase/financials/financials.db). If you haven't run this
ingest before, see codebase/financials/debug_explore for how to do so.
This suite does NOT mock the financials DB - it deliberately runs against
real, already-ingested data, since that is the actual source of truth the
real pipeline depends on.
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from codebase.agent.evals.mock_fixtures import install_mocks, install_test_cache_db

_TEST_DB_PATH = os.path.join(_ROOT, "database", "test_eval_mocked.db")

# Install mocks ONCE at module import time, before any pydantic/langchain-
# dependent codebase.agent.* module is imported anywhere below.
_MOCKS = install_mocks()
install_test_cache_db(_TEST_DB_PATH)

from codebase.agent import clarification  # noqa: E402
from codebase.agent import derived_metrics  # noqa: E402
from codebase.agent import followup  # noqa: E402
from codebase.agent import pipeline  # noqa: E402
from codebase.agent.enrichment import enrich_series  # noqa: E402
from codebase.agent.schemas import QueryUnderstanding  # noqa: E402
from codebase.agent.series_tools import fetch_financial_series  # noqa: E402
from codebase.financials import discovery  # noqa: E402

TEST_SYMBOL = "RELIANCE"


def _require_test_company():
    """Skip a test cleanly (not error) if RELIANCE hasn't been ingested,
    rather than failing with a confusing downstream error.
    """
    if discovery.find_company(TEST_SYMBOL) is None:
        raise unittest.SkipTest(
            f"'{TEST_SYMBOL}' not found in financials.db - run ingest_company('{TEST_SYMBOL}') first."
        )


class TestClarificationGate(unittest.TestCase):
    """Stage 2 - deterministic, no mocking needed beyond the real DB."""

    def setUp(self):
        _require_test_company()

    def test_single_year_no_qualifier_expands_to_trailing_default(self):
        qu = QueryUnderstanding(symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2025"])
        resolved = clarification.resolve_query(qu)
        self.assertTrue(resolved.can_proceed)
        self.assertLessEqual(len(resolved.periods), clarification.DEFAULT_TRAILING_YEARS)
        self.assertEqual(resolved.periods[-1], "Mar 2025")

    def test_single_year_only_returns_exactly_one_period(self):
        qu = QueryUnderstanding(symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2025"], single_year_only=True)
        resolved = clarification.resolve_query(qu)
        self.assertEqual(resolved.periods, ["Mar 2025"])

    def test_explicit_multi_year_is_chronological_regardless_of_input_order(self):
        qu = QueryUnderstanding(symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2025", "2023"])
        resolved = clarification.resolve_query(qu)
        self.assertEqual(resolved.periods, ["Mar 2023", "Mar 2025"])

    def test_unresolved_company_stops_immediately(self):
        qu = QueryUnderstanding(symbol=None, ambiguity_reason="I couldn't identify a known company.")
        resolved = clarification.resolve_query(qu)
        self.assertFalse(resolved.can_proceed)
        self.assertIsNotNone(resolved.stop_message)

    def test_company_resolved_but_not_in_db_stops_with_clear_message(self):
        qu = QueryUnderstanding(symbol="DEFINITELY_NOT_INGESTED_XYZ", line_items=["Sales"])
        resolved = clarification.resolve_query(qu)
        self.assertFalse(resolved.can_proceed)

    def test_derived_metrics_pass_through_unchanged(self):
        qu = QueryUnderstanding(symbol=TEST_SYMBOL, derived_metrics=["ROE"], raw_years=["2024"])
        resolved = clarification.resolve_query(qu)
        self.assertEqual(resolved.derived_metrics, ["ROE"])


class TestSeriesToolsAndEnrichment(unittest.TestCase):
    """Stage 4 + deterministic enrichment - against the real financials DB."""

    def setUp(self):
        _require_test_company()

    def test_fetch_across_all_three_statement_tables(self):
        for line_item, expected_table in [
            ("Sales", "statement_profit_loss"),
            ("Borrowings", "statement_balance_sheet"),
            ("Cash from Operating Activity", "statement_cash_flow"),
        ]:
            result = fetch_financial_series(TEST_SYMBOL, line_item, ["Mar 2024"])
            self.assertTrue(result["ok"], f"failed to fetch {line_item}: {result.get('error')}")
            self.assertEqual(result["table"], expected_table)
            self.assertIsNotNone(result["values"]["Mar 2024"])

    def test_missing_period_is_none_not_dropped(self):
        result = fetch_financial_series(TEST_SYMBOL, "Sales", ["Mar 2010", "Mar 2024"])
        self.assertTrue(result["ok"])
        self.assertIsNone(result["values"]["Mar 2010"])
        self.assertIsNotNone(result["values"]["Mar 2024"])

    def test_unknown_line_item_fails_cleanly(self):
        result = fetch_financial_series(TEST_SYMBOL, "Not A Real Line Item", ["Mar 2024"])
        self.assertFalse(result["ok"])

    def test_enrichment_yoy_and_cagr_match_hand_calculation(self):
        result = fetch_financial_series(TEST_SYMBOL, "Sales", ["Mar 2023", "Mar 2024", "Mar 2025"])
        enriched = enrich_series(result, ["Mar 2023", "Mar 2024", "Mar 2025"])
        self.assertIsNotNone(enriched)

        v23, v24, v25 = (result["values"][p] for p in ["Mar 2023", "Mar 2024", "Mar 2025"])
        expected_yoy_24 = (v24 - v23) / abs(v23) * 100.0
        expected_first_to_last = (v25 - v23) / abs(v23) * 100.0

        self.assertAlmostEqual(enriched.points[1].yoy_change_pct, expected_yoy_24, places=6)
        self.assertAlmostEqual(enriched.first_to_last_change_pct, expected_first_to_last, places=6)
        self.assertEqual(enriched.direction, "up" if expected_first_to_last > 0.5 else "down")

    def test_cagr_is_none_across_a_sign_change(self):
        fake_result = {
            "ok": True, "symbol": "TEST", "line_item": "Net Profit", "unit": "INR_CRORE",
            "table": "statement_profit_loss",
            "values": {"Mar 2022": -500.0, "Mar 2023": 800.0},
        }
        enriched = enrich_series(fake_result, ["Mar 2022", "Mar 2023"])
        self.assertIsNone(enriched.cagr_pct, "CAGR must be None across a loss-to-profit sign change")


class TestDerivedMetrics(unittest.TestCase):
    """ROE / Net Profit Margin / Debt-to-Equity / FCF Margin - real data."""

    def setUp(self):
        _require_test_company()

    def test_net_profit_margin_matches_hand_calculation(self):
        metric = derived_metrics.compute_derived_metric("net_profit_margin", TEST_SYMBOL, ["Mar 2023"])
        net_profit = fetch_financial_series(TEST_SYMBOL, "Net Profit", ["Mar 2023"])["values"]["Mar 2023"]
        sales = fetch_financial_series(TEST_SYMBOL, "Sales", ["Mar 2023"])["values"]["Mar 2023"]
        self.assertAlmostEqual(metric.points[0].value, net_profit / sales, places=9)

    def test_debt_to_equity_uses_closing_balance_only(self):
        metric = derived_metrics.compute_derived_metric("debt_to_equity", TEST_SYMBOL, ["Mar 2023"])
        borrowings = fetch_financial_series(TEST_SYMBOL, "Borrowings", ["Mar 2023"])["values"]["Mar 2023"]
        ec = fetch_financial_series(TEST_SYMBOL, "Equity Capital", ["Mar 2023"])["values"]["Mar 2023"]
        reserves = fetch_financial_series(TEST_SYMBOL, "Reserves", ["Mar 2023"])["values"]["Mar 2023"]
        self.assertAlmostEqual(metric.points[0].value, borrowings / (ec + reserves), places=9)

    def test_roe_uses_average_of_opening_and_closing_equity(self):
        metric = derived_metrics.compute_derived_metric("roe", TEST_SYMBOL, ["Mar 2023"])
        net_profit = fetch_financial_series(TEST_SYMBOL, "Net Profit", ["Mar 2023"])["values"]["Mar 2023"]
        ec_2023 = fetch_financial_series(TEST_SYMBOL, "Equity Capital", ["Mar 2023"])["values"]["Mar 2023"]
        rs_2023 = fetch_financial_series(TEST_SYMBOL, "Reserves", ["Mar 2023"])["values"]["Mar 2023"]
        ec_2022 = fetch_financial_series(TEST_SYMBOL, "Equity Capital", ["Mar 2022"])["values"]["Mar 2022"]
        rs_2022 = fetch_financial_series(TEST_SYMBOL, "Reserves", ["Mar 2022"])["values"]["Mar 2022"]
        avg_equity = ((ec_2023 + rs_2023) + (ec_2022 + rs_2022)) / 2
        self.assertAlmostEqual(metric.points[0].value, net_profit / avg_equity, places=9)

    def test_roe_falls_back_gracefully_with_no_prior_period(self):
        metric = derived_metrics.compute_derived_metric("roe", TEST_SYMBOL, ["Mar 2022"])
        self.assertIsNotNone(metric.points[0].note, "must explain the closing-equity-only fallback")

    def test_every_derived_metric_is_flagged_as_approximation(self):
        for key in ["net_profit_margin", "roe", "debt_to_equity", "fcf_margin"]:
            metric = derived_metrics.compute_derived_metric(key, TEST_SYMBOL, ["Mar 2024"])
            self.assertTrue(metric.is_approximation)
            self.assertIn("computed", metric.label.lower())

    def test_name_resolution_is_case_insensitive_and_flexible(self):
        self.assertEqual(derived_metrics.resolve_derived_metric_name("ROE"), "roe")
        self.assertEqual(derived_metrics.resolve_derived_metric_name("return on equity"), "roe")
        self.assertEqual(derived_metrics.resolve_derived_metric_name("Debt to Equity"), "debt_to_equity")
        self.assertIsNone(derived_metrics.resolve_derived_metric_name("made up ratio"))


class TestFollowupSuggestor(unittest.TestCase):
    """Stage 7 - deterministic, must only ever reference real data."""

    def setUp(self):
        _require_test_company()

    def test_suggestions_never_include_the_metric_just_asked_about(self):
        followup.suggest_follow_ups(TEST_SYMBOL, ["Sales"], ["Mar 2024"], False)
        other_items = followup._other_available_line_items(TEST_SYMBOL, ["Sales"])
        self.assertNotIn("Sales", other_items)

    def test_unknown_company_returns_no_suggestions(self):
        self.assertEqual(followup.suggest_follow_ups("NOT_A_REAL_COMPANY", ["Sales"], ["Mar 2024"], False), [])

    def test_at_most_three_suggestions(self):
        suggestions = followup.suggest_follow_ups(TEST_SYMBOL, ["Sales"], ["Mar 2024"], False)
        self.assertLessEqual(len(suggestions), followup.MAX_SUGGESTIONS)


class TestStructuredCacheKey(unittest.TestCase):
    """The NEW additive build_structured_cache_key method only - the
    original build_cache_key is covered separately in
    test_cachememory_regression.py and is NOT touched by this class.
    """

    def setUp(self):
        from codebase.agentmemory import CacheMemory
        self.cache = CacheMemory(db_path=_TEST_DB_PATH)

    def test_identical_structure_produces_identical_key(self):
        key1, _ = self.cache.build_structured_cache_key(
            company="X", extra_filters={"line_items": ["Sales"], "periods": ["Mar 2023"]}
        )
        key2, _ = self.cache.build_structured_cache_key(
            company="X", extra_filters={"line_items": ["Sales"], "periods": ["Mar 2023"]}
        )
        self.assertEqual(key1, key2)

    def test_different_structure_produces_different_key(self):
        key1, _ = self.cache.build_structured_cache_key(
            company="X", extra_filters={"line_items": ["Sales"], "periods": ["Mar 2023"]}
        )
        key2, _ = self.cache.build_structured_cache_key(
            company="X", extra_filters={"line_items": ["Net Profit"], "periods": ["Mar 2023"]}
        )
        self.assertNotEqual(key1, key2)

    def test_payload_includes_a_question_placeholder_for_schema_compatibility(self):
        _, payload = self.cache.build_structured_cache_key(company="X")
        self.assertIn("question", payload)
        self.assertIsNotNone(payload["question"])


class TestPipelineOrchestration(unittest.TestCase):
    """Full orchestrator wiring, with QU/Synthesis agents mocked via
    mock_fixtures (see module docstring for exactly what is and isn't real).
    """

    def setUp(self):
        _require_test_company()
        # Reset trackers between tests so call counts are test-local.
        _MOCKS["qu_call_count"]["n"] = 0
        _MOCKS["synth_call_count"]["n"] = 0
        _MOCKS["synth_prompts_seen"].clear()
        _MOCKS["qu_responses"].clear()
        _MOCKS["chroma_search_results"].clear()
        # Each test method gets a FRESH cache DB (not the shared
        # _TEST_DB_PATH used by import-time module setup), since several
        # tests below deliberately reuse the same symbol/years and would
        # otherwise collide with cache entries written by a different
        # test method that happened to run first. pipeline.py's
        # module-level _cache singleton is repointed per test.
        import codebase.agent.pipeline as pipeline_module
        from codebase.agentmemory import CacheMemory

        test_db_path = os.path.join(_ROOT, "database", f"test_eval_{self._testMethodName}.db")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        pipeline_module._cache = CacheMemory(db_path=test_db_path)

    def test_cache_miss_then_hit_with_different_phrasing(self):
        _MOCKS["set_qu_response"](
            "Sales for Reliance",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2023"],
            needs_qualitative_context=False, intent="financial",
        )
        _MOCKS["set_qu_response"](
            "What was the sales figure for Reliance",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2023"],
            needs_qualitative_context=False, intent="financial",
        )

        result1 = pipeline.answer_query("Sales for Reliance")
        self.assertFalse(result1["from_cache"])
        self.assertEqual(_MOCKS["synth_call_count"]["n"], 1)

        result2 = pipeline.answer_query("What was the sales figure for Reliance")
        self.assertTrue(result2["from_cache"], "differently-worded but same-structure query must hit cache")
        self.assertEqual(_MOCKS["synth_call_count"]["n"], 1, "synthesis must not run again on a cache hit")
        self.assertEqual(result1["answer"], result2["answer"])

    def test_qualitative_context_changes_cache_key(self):
        _MOCKS["set_qu_response"](
            "Sales no context",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"],
            needs_qualitative_context=False, intent="financial",
        )
        _MOCKS["set_qu_response"](
            "Sales with context",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"],
            needs_qualitative_context=True, intent="financial",
        )
        pipeline.answer_query("Sales no context")
        pipeline.answer_query("Sales with context")
        self.assertEqual(_MOCKS["synth_call_count"]["n"], 2, "different needs_qualitative_context must be separate cache entries")

    def test_derived_metric_only_query_does_not_hit_empty_series_stop(self):
        _MOCKS["set_qu_response"](
            "ROE for Reliance",
            symbol=TEST_SYMBOL, line_items=[], derived_metrics=["ROE"], raw_years=["2023"],
            needs_qualitative_context=False, intent="financial",
        )
        result = pipeline.answer_query("ROE for Reliance")
        self.assertFalse(result["stopped"])
        self.assertIn("computed, average equity", _MOCKS["synth_prompts_seen"][-1])

    def test_unresolvable_company_stops_without_any_agent_call(self):
        result = pipeline.answer_query("complete gibberish query asdkjasdkj")
        self.assertTrue(result["stopped"])
        self.assertEqual(_MOCKS["synth_call_count"]["n"], 0)

    def test_synthesis_prompt_never_contains_unrequested_line_items(self):
        _MOCKS["set_qu_response"](
            "Just sales please",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"],
            needs_qualitative_context=False, intent="financial",
        )
        pipeline.answer_query("Just sales please")
        prompt = _MOCKS["synth_prompts_seen"][-1]
        # Borrowings was NOT asked about - must not appear as a data section
        # (it could theoretically appear in passing English, but should not
        # appear as a "Line item: Borrowings" data block).
        self.assertNotIn("Line item: Borrowings", prompt)


class TestMemoryFeatures(unittest.TestCase):
    """Short-term state, long-term preferences, long-term aliases - added
    on top of the original pipeline. See conversation_state.py,
    agentmemory/preferences.py, financials/aliases.py.
    """

    def setUp(self):
        _require_test_company()
        _MOCKS["qu_call_count"]["n"] = 0
        _MOCKS["synth_call_count"]["n"] = 0
        _MOCKS["synth_prompts_seen"].clear()
        _MOCKS["qu_responses"].clear()
        _MOCKS["chroma_search_results"].clear()

        import codebase.agent.pipeline as pipeline_module
        from codebase.agentmemory import CacheMemory
        from codebase.agentmemory.preferences import UserPreferences
        from codebase.financials import aliases as aliases_module

        test_db_path = os.path.join(_ROOT, "database", f"test_eval_{self._testMethodName}.db")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        pipeline_module._cache = CacheMemory(db_path=test_db_path)
        pipeline_module._preferences = UserPreferences(db_path=test_db_path)
        self.pipeline = pipeline_module
        self.aliases = aliases_module
        self.aliases.init_aliases_schema()

    def tearDown(self):
        # Clean up any alias this test may have written, so tests never
        # leak state into the real financials.db between runs.
        for alias_text in ["test alias xyz", "another test alias"]:
            self.aliases.delete_alias(alias_text)

    def test_conversation_state_round_trips_through_a_resolved_query(self):
        from codebase.agent.clarification import resolve_query
        from codebase.agent.conversation_state import ConversationState
        from codebase.agent.schemas import QueryUnderstanding

        qu = QueryUnderstanding(symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"])
        resolved = clarification.resolve_query(qu)
        state = ConversationState.from_resolved_query(resolved)
        self.assertEqual(state.symbol, TEST_SYMBOL)
        self.assertIn("Sales", state.line_items)
        self.assertFalse(state.is_empty())

    def test_empty_conversation_state_renders_as_no_prior_context(self):
        from codebase.agent.conversation_state import ConversationState
        state = ConversationState()
        self.assertTrue(state.is_empty())
        self.assertIn("first question", state.as_prompt_context().lower())

    def test_followup_query_inherits_symbol_from_prior_state(self):
        from codebase.agent.conversation_state import ConversationState

        prior_state = ConversationState(symbol=TEST_SYMBOL, line_items=["Sales"], periods=["Mar 2024"])

        _MOCKS["set_qu_response"](
            f"{prior_state.as_prompt_context()}\n\nNew question: what about net profit?",
            symbol=TEST_SYMBOL, line_items=["Net Profit"], raw_years=["2024"],
            needs_qualitative_context=False, intent="financial",
        )
        result = self.pipeline.answer_query("what about net profit?", prior_state=prior_state)
        self.assertEqual(result["symbol"], TEST_SYMBOL)
        self.assertEqual(result["line_items"], ["Net Profit"])

    def test_stopped_turn_preserves_prior_state_unchanged(self):
        from codebase.agent.conversation_state import ConversationState

        prior_state = ConversationState(symbol=TEST_SYMBOL, line_items=["Sales"], periods=["Mar 2024"])
        _MOCKS["set_qu_response"](
            "gibberish unresolvable query",
            ambiguity_reason="could not resolve",
        )
        result = self.pipeline.answer_query("gibberish unresolvable query", prior_state=prior_state)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["state"].symbol, TEST_SYMBOL, "prior_state must be preserved, not wiped, on a stop")

    def test_trailing_years_preference_overrides_default(self):
        from codebase.agentmemory.preferences import PreferenceKeys

        self.pipeline._preferences.set_preference(PreferenceKeys.TRAILING_YEARS, 1)
        _MOCKS["set_qu_response"](
            "Sales single year preference test",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"],
            needs_qualitative_context=False, intent="financial",
        )
        result = self.pipeline.answer_query("Sales single year preference test")
        self.assertEqual(len(result["periods"]), 1, "trailing_years=1 preference should yield exactly 1 period")

    def test_always_include_qualitative_preference_forces_context_stage(self):
        from codebase.agentmemory.preferences import PreferenceKeys

        self.pipeline._preferences.set_preference(PreferenceKeys.ALWAYS_INCLUDE_QUALITATIVE, True)

        retrieve_context_calls = []
        original_retrieve_context = self.pipeline.retrieve_context

        def spy_retrieve_context(*args, **kwargs):
            retrieve_context_calls.append((args, kwargs))
            return original_retrieve_context(*args, **kwargs)

        self.pipeline.retrieve_context = spy_retrieve_context
        try:
            _MOCKS["set_qu_response"](
                "Sales without explicit why",
                symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"],
                needs_qualitative_context=False,  # Stage 1 did NOT detect a "why" - preference should override
                intent="financial",
            )
            self.pipeline.answer_query("Sales without explicit why")
        finally:
            self.pipeline.retrieve_context = original_retrieve_context

        self.assertEqual(
            len(retrieve_context_calls), 1,
            "preference should have forced needs_qualitative_context=True, causing Stage 5 to actually run "
            "(even though it may legitimately find zero snippets, which is a separate, valid outcome)",
        )

    def test_preference_never_overrides_explicit_single_year_only(self):
        from codebase.agentmemory.preferences import PreferenceKeys

        self.pipeline._preferences.set_preference(PreferenceKeys.TRAILING_YEARS, 5)
        _MOCKS["set_qu_response"](
            "Just 2024, nothing else",
            symbol=TEST_SYMBOL, line_items=["Sales"], raw_years=["2024"], single_year_only=True,
            needs_qualitative_context=False, intent="financial",
        )
        result = self.pipeline.answer_query("Just 2024, nothing else")
        self.assertEqual(result["periods"], ["Mar 2024"], "explicit single_year_only must win over a trailing_years preference")

    def test_alias_is_learned_on_confident_resolution_and_reused(self):
        result = self.aliases.save_alias("test alias xyz", TEST_SYMBOL, source="llm_confident", source_query="test")
        self.assertTrue(result)
        self.assertEqual(self.aliases.resolve_alias("test alias xyz"), TEST_SYMBOL)

    def test_alias_save_is_a_noop_for_unknown_company(self):
        result = self.aliases.save_alias("another test alias", "DEFINITELY_NOT_A_REAL_SYMBOL")
        self.assertFalse(result)

    def test_alias_save_does_not_overwrite_an_existing_alias(self):
        self.aliases.save_alias("test alias xyz", TEST_SYMBOL, source="manual")
        # Attempt to overwrite with a different (bogus) symbol - must be a no-op
        overwritten = self.aliases.save_alias("test alias xyz", "SOME_OTHER_SYMBOL", source="llm_confident")
        self.assertFalse(overwritten)
        self.assertEqual(self.aliases.resolve_alias("test alias xyz"), TEST_SYMBOL, "original alias must be unchanged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
