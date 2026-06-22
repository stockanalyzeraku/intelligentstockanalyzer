"""CLI entry point for the company Q&A agent.

Run from the project root:

    python codebase/agent/runner.py

Flow per query:
    1. classify_intent() deterministically extracts company/year and decides
       financial vs general intent. No LLM call happens here.
    2. If company or year couldn't be resolved, short-circuit with a
       clarification message - no agent is invoked.
    3. Otherwise invoke financial_agent (financial intent) or general_agent
       (everything else), passing the resolved symbol/period in the message
       so the model never has to guess them itself.
"""
from __future__ import annotations

import logging
import os
import sys
# Allow `python codebase/agent/runner.py` to find the root-level config.py
# and the codebase package regardless of the current working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from codebase.agent.agents import financial_agent, general_agent  # noqa: E402
from codebase.agent.classify import classify_intent  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def answer_query(query: str) -> str:
    """Classify a query and route it to the appropriate agent."""

    result = classify_intent(query)

    if result.unresolved_reason is not None:
        return result.unresolved_reason

    context = f"[Resolved context: symbol={result.symbol}, period={result.period}]"
    user_message = f"{context}\n\nUser question: {query}"

    agent = financial_agent if result.intent == "financial" else general_agent
    logger.info(
        "Routing query to %s agent (symbol=%s, period=%s)",
        result.intent,
        result.symbol,
        result.period,
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    return response["messages"][-1].content_blocks

def main() -> None:
    """Run an interactive CLI loop for ad-hoc testing."""
    print("Company Q&A agent (POC). Type 'exit' to quit.")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        try:
            answer = answer_query(query)
        except Exception:  # noqa: BLE001 - top-level CLI guard for the POC
            logger.exception("Unhandled error answering query: %r", query)
            print("Sorry, something went wrong answering that question.")
            continue
        print(answer)


if __name__ == "__main__":
    main()
