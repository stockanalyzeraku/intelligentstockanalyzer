from __future__ import annotations
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Optional

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

_CHILD_COLLECTION   = CONFIG.COL_CHILD
_PARENT_COLLECTION  = CONFIG.COL_PARENT
_DISTANCE_THRESHOLD = 0.75

# ------------------------------------------------------------------ #
#  ChromaStore instance                                                #
# ------------------------------------------------------------------ #

_store = ChromaStore.get_instance()

# ------------------------------------------------------------------ #
#  Tool helpers                                                        #
# ------------------------------------------------------------------ #

def _format_results(results: list[dict]) -> str:
    """Formats parent-expanded retrieval results into readable text for the LLM."""
    if not results:
        return "INSUFFICIENT_EVIDENCE: no relevant parent chunks were retrieved."

    output = []
    for index, item in enumerate(results, start=1):
        parent_meta = item.get("parent_metadata", {})
        child_meta = item.get("child_metadata", {})
        dist = item.get("distance", 1.0)
        if dist > _DISTANCE_THRESHOLD:
            continue
        output.append(
            f"[Source {index} | page={parent_meta.get('page_number', '?')} | "
            f"section={parent_meta.get('section', 'unknown')} | "
            f"parent_id={item.get('parent_id', 'unknown')} | "
            f"child_id={item.get('child_id', 'unknown')} | "
            f"child_index={child_meta.get('child_index', '?')} | "
            f"collections={_CHILD_COLLECTION} -> {_PARENT_COLLECTION} | "
            f"relevance={1 - dist:.2f}]\n"
            f"{item.get('parent_text', '')}"
        )

    if not output:
        return "INSUFFICIENT_EVIDENCE: retrieved chunks did not clear the relevance threshold."

    return "\n\n---\n\n".join(output)


def _log_retrieval_bundle(query: str, results: list[dict], where: dict | None = None) -> None:
    """Log which child and parent chunks were used for one retrieval step."""
    if not results:
        logger.info(
            "[Agent] Retrieval bundle empty",
            query=query,
            where=where,
        )
        return

    logger.info(
        "[Agent] Retrieval bundle",
        query=query,
        where=where,
        result_count=len(results),
        parent_ids=[item.get("parent_id") for item in results],
        child_ids=[item.get("child_id") for item in results],
    )

    for item in results:
        parent_meta = item.get("parent_metadata", {})
        child_meta = item.get("child_metadata", {})
        logger.debug(
            "[Agent] Retrieval item",
            parent_id=item.get("parent_id"),
            child_id=item.get("child_id"),
            parent_page=parent_meta.get("page_number"),
            parent_section=parent_meta.get("section"),
            child_index=child_meta.get("child_index"),
            distance=item.get("distance"),
            parent_preview=(item.get("parent_text", "")[:240]),
            child_preview=(item.get("child_text", "")[:240]),
        )


def _safe_query(query: str, n_results: int = 6, where: dict | None = None) -> str:
    """Search child chunks, then expand to the best parent chunks."""
    try:
        results = _store.query_children_with_parent_context(
            query_texts = [query],
            n_results   = n_results,
            where       = where,
        )
        _log_retrieval_bundle(query, results, where)
        formatted = _format_results(results)
        if formatted.startswith("INSUFFICIENT_EVIDENCE"):
            return formatted
        return formatted
    except Exception as e:
        logger.warning(f"[Agent] Query failed ({e}), retrying without filter.")
        try:
            results = _store.query_children_with_parent_context(
                query_texts = [query],
                n_results   = n_results,
            )
            _log_retrieval_bundle(query, results)
            formatted = _format_results(results)
            if formatted.startswith("INSUFFICIENT_EVIDENCE"):
                return formatted
            return formatted
        except Exception as exc:
            return f"Search failed: {exc}"


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
            results = _store.query_children_with_parent_context(
                query_texts = [actual_query],
                n_results   = 8,
                where       = where,
            )
            _log_retrieval_bundle(actual_query, results, where)
            formatted = _format_results(results)
            if not formatted.startswith("INSUFFICIENT_EVIDENCE"):
                return formatted
            logger.warning("[Tool] No sufficiently relevant results with filters, falling back to general search.")
        except Exception as e:
            logger.warning(f"[Tool] Filtered search failed ({e}), falling back to general search.")

    return _safe_query(actual_query, n_results=8)


# ------------------------------------------------------------------ #
#  Tools list                                                          #
# ------------------------------------------------------------------ #

TOOLS = [
    search_annual_report,
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
        "What are the fixed assets bought by Kalyan Jewellers?",
        "Give me details of financial numbers for 2025",
        "Give me details between 2023 and 2025 financial performance"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = agent.ask(q)
        print(f"A: {answer}")
        print()