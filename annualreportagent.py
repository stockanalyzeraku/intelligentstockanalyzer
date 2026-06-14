from __future__ import annotations
import os
import sys
from unittest import result

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
from typing import Any, Optional

from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import CONFIG
from logger import get_logger
from codebase.vectordb.chromastore import ChromaStore

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

_COLLECTION        = "kalyan_annual_2025"
_DISTANCE_THRESHOLD = 0.75

# ------------------------------------------------------------------ #
#  ChromaStore instance                                                #
# ------------------------------------------------------------------ #

_store = ChromaStore.get_instance()

# ------------------------------------------------------------------ #
#  Tool helpers                                                        #
# ------------------------------------------------------------------ #

def _format_results(results: dict) -> str:
    """Formats ChromaDB results into readable text for the LLM."""
    ids       = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    if not ids:
        return "No relevant information found in the document."

    output = []
    for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        if dist > _DISTANCE_THRESHOLD:
            continue
        output.append(
            f"[Page {meta.get('page_number', '?')} | "
            f"Section: {meta.get('section', 'unknown')} | "
            f"Relevance: {1 - dist:.2f}]\n{text}"
        )

    return "\n\n---\n\n".join(output) if output else "No relevant information found."


def _safe_query(query: str, n_results: int = 6, where: dict | None = None) -> str:
    """Runs a ChromaDB query with optional filter, falls back without filter on error."""
    try:
        results = _store.query_collection(
            collection_name = _COLLECTION,
            query_texts     = [query],
            n_results       = n_results,
            where           = where,
        )
        return _format_results(results)
    except Exception as e:
        if where:
            logger.warning(f"[Agent] Query with filter failed ({e}), retrying without filter.")
            try:
                results = _store.query_collection(
                    collection_name = _COLLECTION,
                    query_texts     = [query],
                    n_results       = n_results,
                )
                return _format_results(results)
            except Exception as e2:
                return f"Search failed: {e2}"
        return f"Search failed: {e}"


# ------------------------------------------------------------------ #
#  Tool functions                                                      #
# ------------------------------------------------------------------ #
@tool
def search_annual_report(query: str) -> str:
    """
    Searches the Kalyan Jewellers Annual Report with optional filters.
    Use this for any question about the company.
    
    Input formats:
    - Plain query:                    "what is the revenue for FY25?"
    - Filter by intent:               "intent:financial_performance | revenue FY25"
    - Filter by page:                 "page:14 | what is on this page?"
    - Filter by year:                 "year:2025 | revenue growth"
    - Filter by has_table:            "has_table:true | show me all tables"
    - Multiple filters:               "intent:company_overview,year:2025 | brief of company"

    Available intents:
    table_of_contents, company_overview, performance_highlights,
    strategic_overview, esg_csr, general_narrative, financial_performance,
    management_commentary, operational_highlights, risk_factors,
    corporate_governance, auditor_report, timeline
    """
    logger.info(f"[Tool] search_annual_report — input='{query}'")

    # ── Parse filters and query ────────────────────────────────────────
    where        : dict  = {}
    actual_query : str   = query

    if "|" in query:
        filter_str, actual_query = [p.strip() for p in query.split("|", 1)]

        for token in filter_str.split(","):
            token = token.strip()

            if token.startswith("intent:"):
                where["page_intent"] = {"$contains": token.replace("intent:", "").strip()}

            elif token.startswith("year:"):
                try:
                    where["year"] = int(token.replace("year:", "").strip())
                except ValueError:
                    pass

            elif token.startswith("page:"):
                try:
                    where["page_number"] = int(token.replace("page:", "").strip())
                except ValueError:
                    pass

            elif token.startswith("has_table:"):
                val = token.replace("has_table:", "").strip().lower()
                where["has_table"] = val == "true"

            elif token.startswith("section:"):
                where["section"] = token.replace("section:", "").strip()

            elif token.startswith("doc_type:"):
                where["doc_type"] = token.replace("doc_type:", "").strip()

    logger.info(f"[Tool] filters={where} | query='{actual_query}'")

    # ── Search with filters ────────────────────────────────────────────
    if where:
        try:
            results   = _store.query_collection(
                collection_name = _COLLECTION,
                query_texts     = [actual_query],
                n_results       = 8,
                where           = where,
            )
            formatted = _format_results(results)
            if "No relevant information" not in formatted:
                return formatted
            logger.warning("[Tool] No results with filters, falling back to general search.")
        except Exception as e:
            logger.warning(f"[Tool] Filtered search failed ({e}), falling back to general search.")

    # ── Fallback — no filters ──────────────────────────────────────────
    return _safe_query(actual_query, n_results=8)

# def search_annual_report(query: str) -> str:
#     """
#     Searches the Kalyan Jewellers Annual Report 2025 for relevant information.
#     Use this for any general question about the company, operations, strategy, or overview.
#     """
#     logger.info(f"[Tool] search_annual_report — query='{query}'")
#     return _safe_query(query, n_results=6)


# def search_financial_data(query: str) -> str:
#     """
#     Searches for financial numbers, tables, revenue, profit, assets, ratios,
#     fixed assets, capital expenditure, and any numeric financial data from
#     the Kalyan Jewellers Annual Report 2025.
#     Use this for any question involving numbers or financial figures.
#     """
#     logger.info(f"[Tool] search_financial_data — query='{query}'")
#     return _safe_query(query, n_results=8, where={"has_table": True})


# def search_management_commentary(query: str) -> str:
#     """
#     Searches for management commentary, MD&A, chairman letter, CEO statements,
#     and strategic direction from leadership in the Kalyan Jewellers Annual Report 2025.
#     Use this when asked what management or MD said about anything.
#     """
#     logger.info(f"[Tool] search_management_commentary — query='{query}'")
#     return _safe_query(query, n_results=6, where={"section": "management"})


# def search_timeline(query: str) -> str:
#     """
#     Searches for timeline, milestones, history, and chronological events
#     from the Kalyan Jewellers Annual Report 2025.
#     Use this for questions about company history or a timeline of events.
#     """
#     logger.info(f"[Tool] search_timeline — query='{query}'")
#     return _safe_query(query, n_results=6, where={"section": "timeline"})


def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression for financial calculations.
    Use this to calculate growth percentages, ratios, or comparisons between numbers.
    Input must be a valid Python math expression.
    Example input: (21092 - 16600) / 16600 * 100
    """
    logger.info(f"[Tool] calculate — expression='{expression}'")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result:.4f}"
    except Exception as e:
        return f"Calculation failed: {e}"


def get_chunk_by_id(chunk_id: str) -> str:
    """
    Fetches a specific chunk by its exact ID from the annual report.
    Use this when you already know the chunk ID and need its full content.
    Example input: page_16_text_chunk_3
    """
    logger.info(f"[Tool] get_chunk_by_id — id='{chunk_id}'")
    try:
        chunk = _store.get_by_id(_COLLECTION, chunk_id)
        if chunk:
            return (
                f"ID       : {chunk['id']}\n"
                f"Metadata : {chunk['metadata']}\n"
                f"Text     : {chunk['document']}"
            )
        return f"Chunk '{chunk_id}' not found."
    except Exception as e:
        return f"get_by_id failed: {e}"


# ------------------------------------------------------------------ #
#  Tools list                                                          #
# ------------------------------------------------------------------ #

TOOLS = [
    search_annual_report,
    calculate,
    get_chunk_by_id,
]

# ------------------------------------------------------------------ #
#  Agent                                                               #
# ------------------------------------------------------------------ #

class AnnualReportAgent:
    """
    ReAct agent that answers questions about Kalyan Jewellers Annual Report 2025.
    Primary LLM  : Mistral medium-3
    Fallback LLM : Google Gemini 1.5 Pro (used only when Mistral API fails)

    Context window decisions
    ------------------------
    - General search   : 6 results (~2400 tokens)
    - Financial search : 8 results (~3200 tokens)
    - Distance cutoff  : 0.75 (cosine) — anything above is noise
    - Max iterations   : 6 — enough for multi-step, prevents loops
    """

    def __init__(self):
        self._mistral       = self._build_mistral()
        self._gemini        = self._build_gemini()
        self._agent_mistral = create_react_agent(model=self._mistral, tools=TOOLS)
        self._agent_gemini  = create_react_agent(model=self._gemini,  tools=TOOLS)
        logger.info("[AnnualReportAgent] Ready — primary=Mistral, fallback=Gemini")

    # ------------------------------------------------------------------ #
    #  LLM builders                                                        #
    # ------------------------------------------------------------------ #

    def _build_mistral(self) -> ChatMistralAI:
        return ChatMistralAI(
            model       = "open-mistral-nemo",
            api_key     = CONFIG.MISTRAL_API_KEY,
            temperature = 0,
        )

    def _build_gemini(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model          = CONFIG.GEMINI_MODEL,
            google_api_key = CONFIG.GEMINI_API_KEY,
            temperature    = 0,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        """
        Ask a question about the Kalyan Jewellers Annual Report 2025.
        Tries Mistral first, falls back to Gemini if Mistral API fails.

        Parameters
        ----------
        question : Natural language question.

        Returns
        -------
        str : Agent's final answer.
        """
        logger.info(f"[AnnualReportAgent] Question: {question}")

        # ── Primary: Mistral ───────────────────────────────────────────
        try:
            logger.info("[AnnualReportAgent] Using Mistral")
            result = self._agent_mistral.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            answer = result["messages"][-1].content
            logger.info("[AnnualReportAgent] Mistral answered successfully")
            return answer

        except Exception as mistral_err:
            logger.warning(
                f"[AnnualReportAgent] Mistral failed — {mistral_err}. "
                f"Falling back to Gemini."
            )

        # ── Fallback: Gemini ───────────────────────────────────────────
        try:
            logger.info("[AnnualReportAgent] Using Gemini fallback")
            result = self._agent_gemini.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
#            answer = result["messages"][-1].content
            raw = result["messages"][-1].content
            if isinstance(raw, list):
                answer = " ".join(block["text"] for block in raw if isinstance(block, dict) and "text" in block
                )
            else:
                answer = raw
            logger.info("[AnnualReportAgent] Gemini answered successfully")
            return answer

        except Exception as gemini_err:
            logger.error(f"[AnnualReportAgent] Gemini also failed — {gemini_err}")
            return (
                f"Both LLMs failed.\n"
            #    f"Mistral error : {mistral_err}\n"
                f"Gemini error  : {gemini_err}"
            )


# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    agent = AnnualReportAgent()

    questions = [
        "Give me a brief of the 2025 Annual Report",
        "What did the MD say about the annual report this year?",
        "Give me the complete timeline of Kalyan Jewellers",
        "What are the fixed assets bought by Kalyan?",
        "Give me details of financial numbers for 2025",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = agent.ask(q)
        print(f"A: {answer}")
        print()