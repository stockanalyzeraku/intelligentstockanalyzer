"""Real-API eval: spot-checks actual Mistral model behavior, occasionally.

UNLIKE test_pipeline_mocked.py, this makes REAL API calls (Query
Understanding + Synthesis agents) against whatever model is configured in
agents.py / query_understanding.py / synthesis.py - it costs real money
and needs a real MISTRAL_API_KEY available. Run this occasionally, not on
every commit, specifically to catch prompt drift that mocked tests cannot:
does the real model actually extract the right structured fields, and
does it actually follow the synthesis rules (no model arithmetic, always
label derived metrics as approximations, never mention metrics that
weren't asked about, no peer comparison)?

USAGE (from project root, with MISTRAL_API_KEY set in your environment):

    python codebase/agent/evals/eval_real_api.py

    # Spot-check a specific company instead of the default:
    python codebase/agent/evals/eval_real_api.py --symbol KALYANKJIL

Each case prints PASS/FAIL/WARN per check, plus the raw model output, so
you can read the actual answer text yourself - some checks here (e.g.
"does the answer avoid mentioning unrelated metrics") are necessarily
heuristic substring checks, not a guarantee, and are flagged as WARN
rather than FAIL when a check is inherently fuzzy.
"""

from __future__ import annotations

import argparse
import os
import sys


def _check_api_key_present() -> bool:
    try:
        from config import CONFIG
        return bool(getattr(CONFIG, "MISTRAL_API_KEY", None))
    except Exception:
        return False


CASES = [
    {
        "name": "single_year_extraction",
        "query": "What was the Sales for Reliance in 2023",
        "checks": [
            ("symbol_resolved", lambda qu: qu.symbol == "RELIANCE"),
            ("line_item_is_sales_only", lambda qu: qu.line_items == ["Sales"]),
            ("year_extracted", lambda qu: qu.raw_years == ["2023"]),
            ("no_derived_metrics", lambda qu: qu.derived_metrics == []),
            ("not_flagged_qualitative", lambda qu: qu.needs_qualitative_context is False),
        ],
    },
    {
        "name": "derived_metric_extraction",
        "query": "What is Reliance's ROE for 2024",
        "checks": [
            ("symbol_resolved", lambda qu: qu.symbol == "RELIANCE"),
            ("roe_in_derived_metrics", lambda qu: any("roe" in d.lower() for d in qu.derived_metrics)),
            ("roe_not_in_line_items", lambda qu: not any("roe" in li.lower() for li in qu.line_items)),
        ],
    },
    {
        "name": "qualitative_intent_detected",
        "query": "Why did Reliance's sales grow in 2024?",
        "checks": [
            ("symbol_resolved", lambda qu: qu.symbol == "RELIANCE"),
            ("flagged_qualitative", lambda qu: qu.needs_qualitative_context is True),
        ],
    },
    {
        "name": "unresolvable_company",
        "query": "What was the revenue of Zorptech Industries in 2023",
        "checks": [
            ("symbol_not_guessed", lambda qu: qu.symbol is None),
            ("ambiguity_reason_set", lambda qu: qu.ambiguity_reason is not None),
        ],
    },
    {
        "name": "no_unrequested_metric_scope_creep",
        "query": "Just tell me Sales for 2024, nothing else",
        "checks": [
            ("symbol_resolved", lambda qu: qu.symbol == "RELIANCE"),
            ("only_sales_requested", lambda qu: qu.line_items == ["Sales"]),
        ],
    },
]


def run_query_understanding_checks(symbol_override):
    from codebase.agent.query_understanding import understand_query

    results = []
    for case in CASES:
        query = case["query"]
        if symbol_override:
            query = query.replace("Reliance", symbol_override).replace("Zorptech Industries", symbol_override)
        qu = understand_query(query)
        case_result = {"name": case["name"], "query": query, "qu": qu, "checks": []}
        for check_name, check_fn in case["checks"]:
            try:
                passed = check_fn(qu)
            except Exception as exc:  # noqa: BLE001
                passed = False
                check_name = f"{check_name} (raised {exc!r})"
            case_result["checks"].append((check_name, passed))
        results.append(case_result)
    return results


def run_synthesis_checks(symbol):
    """Spot-check Stage 6 (Synthesis) directly with hand-built inputs, so
    this doesn't depend on Stage 4/5 also working - isolates whether the
    SYNTHESIS PROMPT specifically is being followed.
    """
    from codebase.agent.derived_metrics import compute_derived_metric
    from codebase.agent.enrichment import enrich_series
    from codebase.agent.series_tools import fetch_financial_series
    from codebase.agent.synthesis import synthesize_answer
    from codebase.financials import discovery

    if discovery.find_company(symbol) is None:
        return [{"name": "synthesis_checks", "skipped": True, "reason": f"'{symbol}' not ingested"}]

    periods = ["Mar 2023", "Mar 2024", "Mar 2025"]
    fetch_result = fetch_financial_series(symbol, "Sales", periods)
    if not fetch_result["ok"]:
        return [{"name": "synthesis_checks", "skipped": True, "reason": fetch_result.get("error")}]
    enriched = enrich_series(fetch_result, periods)

    roe_metric = compute_derived_metric("roe", symbol, periods)

    answer = synthesize_answer(
        f"Sales trend and ROE for {symbol}",
        symbol,
        [enriched],
        derived_metrics=[roe_metric] if roe_metric else None,
    )

    checks = [
        ("mentions_an_actual_sales_value", any(
            str(int(p.value)) in answer.replace(",", "") for p in enriched.points if p.value is not None
        )),
        ("does_not_mention_unrelated_metrics", not any(
            term in answer.lower() for term in ["borrowings", "reserves", "depreciation", "interest expense"]
        )),
        ("labels_roe_as_computed_or_approximate", any(
            term in answer.lower() for term in ["computed", "approximate", "approximation", "estimated"]
        ) if roe_metric else True),
        ("no_peer_or_industry_comparison", not any(
            term in answer.lower() for term in ["competitor", "industry average", "peers", "compared to other companies"]
        )),
    ]

    return [{"name": "synthesis_checks", "query": f"Sales trend and ROE for {symbol}", "answer": answer, "checks": checks}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="RELIANCE", help="Company symbol to spot-check (default: RELIANCE)")
    args = parser.parse_args()

    if not _check_api_key_present():
        print("ERROR: MISTRAL_API_KEY not configured (check config.py / your environment).")
        print("This script makes real API calls and cannot run without it.")
        sys.exit(1)

    print(f"Running real-API eval against symbol={args.symbol}")
    print("This makes real Mistral API calls and will cost money.\n")

    print("=" * 70)
    print("QUERY UNDERSTANDING (Stage 1) checks")
    print("=" * 70)
    qu_results = run_query_understanding_checks(args.symbol if args.symbol != "RELIANCE" else None)
    total_checks = 0
    total_passed = 0
    for case_result in qu_results:
        print(f"\n[{case_result['name']}] query: {case_result['query']!r}")
        print(f"  extracted: {case_result['qu']}")
        for check_name, passed in case_result["checks"]:
            total_checks += 1
            total_passed += int(passed)
            print(f"    {'PASS' if passed else 'FAIL'} - {check_name}")

    print()
    print("=" * 70)
    print("SYNTHESIS (Stage 6) checks")
    print("=" * 70)
    synth_results = run_synthesis_checks(args.symbol)
    for case_result in synth_results:
        if case_result.get("skipped"):
            print(f"\nSKIPPED - {case_result['reason']}")
            continue
        print(f"\n[{case_result['name']}] query: {case_result['query']!r}")
        print(f"  answer:\n{case_result['answer']}\n")
        for check_name, passed in case_result["checks"]:
            total_checks += 1
            total_passed += int(passed)
            print(f"    {'PASS' if passed else 'WARN (heuristic check)'} - {check_name}")

    print()
    print("=" * 70)
    print(f"SUMMARY: {total_passed}/{total_checks} checks passed")
    print("=" * 70)
    print(
        "Note: synthesis checks are heuristic (substring matching on free-form "
        "text) and may have false positives/negatives - read the actual answer "
        "text above to judge for yourself, especially for any WARN result."
    )


if __name__ == "__main__":
    main()
