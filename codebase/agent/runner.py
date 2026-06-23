"""CLI entry point for the multi-agent company Q&A pipeline.

All queries go through codebase.agent.pipeline.answer_query(), which runs
the full 8-stage pipeline:
    Query Understanding -> Clarification Gate -> Cache check ->
    Data Retrieval+Enrichment -> conditional Context Retrieval ->
    Synthesis -> Follow-up Suggestor -> Cache write

Run from the project root:
    python codebase/agent/runner.py
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from codebase.agent.pipeline import answer_query  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_for_display(result: dict) -> str:
    """Turn the pipeline's structured result dict into CLI-friendly text."""
    lines = [result["answer"]]

    if result.get("suggestions"):
        lines.append("")
        lines.append("You might also ask:")
        for suggestion in result["suggestions"]:
            lines.append(f"  - {suggestion}")

    if result.get("from_cache"):
        lines.append("")
        lines.append("(served from cache)")

    return "\n".join(lines)


def main() -> None:
    """Run an interactive CLI loop for ad-hoc testing."""
    print("Company Q&A multi-agent pipeline. Type 'exit' to quit.")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        try:
            result = answer_query(query)
        except Exception:
            logger.exception("Unhandled error answering query: %r", query)
            print("Sorry, something went wrong answering that question.")
            continue
        print(_format_for_display(result))


if __name__ == "__main__":
    main()
