
# =============================================================================
# CELL 20 — REACT Agent
# =============================================================================
"""
InvestmentBrainAgent: full REACT orchestration loop.
Validate → Understand → Retrieve → Observe → Gap Analyse → Synthesise.
Maintains conversation history of last 5 exchanges per session.
"""

from typing import Dict, List, Optional
from logger import get_logger
from config import CONFIG
from inputvalidator import InputValidator, ValidationError
from workingmemory import WorkingMemory
from gapanalyzer import GAP_ANALYSER
from queryinterpreter import QUERY_ENGINE
from storemanager import STORAGE_MANAGER
from errorhandler import CB_CHROMADB
from chroma import CHROMA_STORE
import os
from synthesisanalyzer import SYNTHESIS_ENGINE


class InvestmentBrainAgent:
    """
    Full REACT agent for investment research over annual reports.

    Entry point: ask(question, scrip=None, fiscal_year=None)

    REACT Loop
    ----------
    1. Validate input
    2. Understand intent (QueryUnderstandingEngine)
    3. Retrieve chunks (HybridRetriever) — up to MAX_REACT_ITERATIONS
    4. Expand parent sections for top chunks
    5. Analyse gaps (GapAnalyser)
    6. If gaps and iterations remain: form gap-targeted queries and retry
    7. Synthesise answer (SynthesisEngine)
    8. Log query to SQLite
    9. Return structured result dict
    """

    def __init__(self):
        """Initialise agent with shared component instances."""
        self._logger = get_logger("agent")
        self._query_engine = QUERY_ENGINE
        self._retriever = RETRIEVER
        self._gap_analyser = GAP_ANALYSER
        self._synthesis_engine = SYNTHESIS_ENGINE
        self._storage = STORAGE_MANAGER
        self._conversation_history: List[Dict] = []

    def ask(
        self,
        question: str,
        scrip: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> Dict:
        """
        Answer an investment research question using the REACT loop.

        Parameters
        ----------
        question : str
            User question.
        scrip : str, optional
            Restrict retrieval to a specific company.
        fiscal_year : str, optional
            Restrict retrieval to a specific fiscal year.

        Returns
        -------
        dict
            Keys: answer, sources, confidence, follow_ups, verified,
                  issues, iterations, gaps.
        """
        # Step 1: Validate
        try:
            question = InputValidator.validate_question(question)
        except ValidationError as exc:
            return {
                "answer": f"Invalid question: {exc}",
                "sources": [], "confidence": 0.0, "follow_ups": [],
                "verified": False, "issues": [str(exc)], "iterations": 0, "gaps": [],
            }

        if scrip:
            try:
                scrip = InputValidator.validate_scrip(scrip)
            except ValidationError as exc:
                self._logger.warning("Invalid scrip ignored", scrip=scrip, error=str(exc))
                scrip = None

        if fiscal_year:
            try:
                fiscal_year = InputValidator.validate_fiscal_year(fiscal_year)
            except ValidationError as exc:
                self._logger.warning("Invalid FY ignored", fy=fiscal_year, error=str(exc))
                fiscal_year = None

        self._logger.info("Agent.ask() called", question=question[:80], scrip=scrip, fy=fiscal_year)

        # Step 2: Understand intent
        intent = self._query_engine.analyse(question)
        eff_scrip = scrip or intent.scrip_filter
        eff_fy = fiscal_year or intent.fy_filter

        # Step 3: Initialise working memory
        memory = WorkingMemory(question=question, intent=intent)

        # Seed tried_queries with all sub-queries
        queries_to_try: List[str] = list(intent.sub_queries) + intent.expanded_terms[:3]
        memory.tried_queries.extend(queries_to_try)

        # REACT loop
        for iteration in range(1, CONFIG.MAX_REACT_ITERATIONS + 1):
            memory.iteration = iteration
            self._logger.info("REACT iteration", iteration=iteration, queries=len(queries_to_try))

            # Step 4: Retrieve
            for q in queries_to_try[:4]:  # cap per-iteration queries
                chunks = self._retriever.retrieve(
                    query=q,
                    collections=intent.target_collections,
                    query_category=intent.category,
                    scrip=eff_scrip,
                    fiscal_year=eff_fy,
                )
                memory.add_chunks(chunks)

            # Step 5: Expand parents for top chunks
            self._expand_parents(memory)

            # Step 6: Gap analysis
            sufficient, gaps = self._gap_analyser.analyse(memory)

            if sufficient or iteration == CONFIG.MAX_REACT_ITERATIONS:
                break

            # Step 7: Form gap-targeted queries for next iteration
            gap_queries = self._gaps_to_queries(gaps, memory)
            # Filter already-tried queries
            queries_to_try = [q for q in gap_queries if q not in memory.tried_queries]
            if not queries_to_try:
                self._logger.info("No new gap queries — stopping REACT loop early.")
                break
            memory.tried_queries.extend(queries_to_try)

        # Step 8: Synthesise
        result = self._synthesis_engine.synthesise(memory)

        # Step 9: Log
        self._storage.log_query(
            question=question,
            scrip=eff_scrip,
            fiscal_year=eff_fy,
            answer_len=len(result.get("answer", "")),
            confidence=result.get("confidence", 0.0),
            iterations=memory.iteration,
            verified=result.get("verified", False),
        )

        # Update conversation history
        self._conversation_history.append({"q": question, "a": result.get("answer", "")[:500]})
        if len(self._conversation_history) > CONFIG.CONVERSATION_HISTORY_LIMIT:
            self._conversation_history.pop(0)

        self._logger.info(
            "Agent.ask() complete",
            iterations=memory.iteration,
            chunks_used=len(memory.retrieved_chunks),
            confidence=result.get("confidence"),
            verified=result.get("verified"),
        )
        return result

    def show_status(self) -> None:
        """Print database statistics and list of processed files."""
        print(CHROMA_STORE.status())
        files = self._storage.get_processed_files()
        print(f"\n=== Processed Files ({len(files)}) ===")
        for f in files:
            print(f"  {f['scrip']} {f['fiscal_year']} — {os.path.basename(f['file_path'])} ({f['chunk_count']} chunks)")

    # ── Private helpers ───────────────────────────────────────────────────

    def _expand_parents(self, memory: WorkingMemory) -> None:
        """
        Fetch parent sections for the top retrieved chunks.

        Only expands top-5 unique parent IDs to control context window size.

        Parameters
        ----------
        memory : WorkingMemory
        """
        seen_parents: set = set()
        for chunk in memory.retrieved_chunks[:10]:
            parent_id = chunk.get("metadata", {}).get("parent_id")
            if not parent_id or parent_id in seen_parents:
                continue
            if parent_id == chunk.get("id"):
                continue  # skip parent-level chunks
            seen_parents.add(parent_id)
            if len(seen_parents) > 5:
                break
            parent = self._retriever.get_parent_context(parent_id)
            if parent:
                memory.add_parent_context(parent)

    def _gaps_to_queries(self, gaps: List[str], memory: WorkingMemory) -> List[str]:
        """
        Convert gap descriptions into targeted retrieval queries.

        Parameters
        ----------
        gaps : List[str]
            Gap descriptions from GapAnalyser.
        memory : WorkingMemory

        Returns
        -------
        List[str]
            New query strings.
        """
        queries: List[str] = []
        for gap in gaps[:3]:
            # Combine gap description with original question context
            q = f"{memory.question} {gap}"
            queries.append(q.strip())
        return queries


# Instantiate the global agent
AGENT = InvestmentBrainAgent()
print("[Cell 20] InvestmentBrainAgent ready. Use AGENT.ask('your question') to query.")

# ----------------------------------------------------------------------------
# Cell 20: REACT Agent
# Purpose: Orchestrate the full REACT loop from question to verified answer.
# Key Classes: InvestmentBrainAgent
# Key Functions:
#   InvestmentBrainAgent.ask(question, scrip, fiscal_year) → dict
#   InvestmentBrainAgent.show_status() → None
#   InvestmentBrainAgent._expand_parents(memory) → None
#   InvestmentBrainAgent._gaps_to_queries(gaps, memory) → List[str]
# Key Constants/Config: CONFIG.MAX_REACT_ITERATIONS, CONFIG.CONVERSATION_HISTORY_LIMIT
# Imports exported: InvestmentBrainAgent, AGENT
# Depends on: Cells 3–19 (all prior cells)
# Critical notes: AGENT is the single global instance — use it for all queries.
#   Conversation history is in-memory and resets on Colab restart.
#   _expand_parents caps at 5 unique parents to prevent context overflow.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

