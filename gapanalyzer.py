
# =============================================================================
# CELL 18 — Gap Analyser
# =============================================================================
"""
GapAnalyser: evaluate whether working memory provides sufficient coverage
to answer the question, and identify specific information gaps.
"""

import json
import re
from typing import List, Tuple
from logger import get_logger
from config import CONFIG
from workingmemory import WorkingMemory
from errorhandler import CircuitBreakerOpenError, CB_GEMINI



_GAP_SYSTEM_PROMPT = """You are a financial research quality assessor.
Given a user question and a summary of retrieved context, determine:
1. Whether the context is SUFFICIENT to answer the question confidently.
2. What specific information is MISSING (gaps).

Return ONLY a JSON object with these keys:
- sufficient: true or false
- gaps: list of strings describing missing information (empty list if sufficient)
- confidence: float 0.0 to 1.0 representing how well the question can be answered

No explanation. No markdown. JSON only."""


class GapAnalyser:
    """
    Assesses working memory completeness and identifies retrieval gaps.

    Falls back to a chunk-count heuristic if the LLM call fails.
    """

    def __init__(self):
        """Initialise LLM client for gap analysis."""
        self._logger = get_logger("gap_analyser")
        self._model = self._init_model()

    def _init_model(self):
        """Initialise Gemini model for gap assessment."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=CONFIG.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=CONFIG.GEMINI_MODEL,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                },
                system_instruction=_GAP_SYSTEM_PROMPT,
            )
        except Exception as exc:
            self._logger.error("Failed to init Gemini for gap analyser", error=str(exc))
            return None

    def analyse(self, memory: WorkingMemory) -> Tuple[bool, List[str]]:
        """
        Evaluate whether current memory is sufficient to answer the question.

        Updates memory.confidence and memory.identified_gaps as side effects.

        Parameters
        ----------
        memory : WorkingMemory
            Current agent working memory.

        Returns
        -------
        Tuple[bool, List[str]]
            (sufficient, list_of_gap_descriptions)
        """
        if self._model is None or not memory.retrieved_chunks:
            return self._heuristic_fallback(memory)

        coverage_summary = self._summarise_coverage(memory)
        prompt = (
            f"Question: {memory.question}\n\n"
            f"Retrieved context summary:\n{coverage_summary}"
        )

        try:
            response = CB_GEMINI.call(
                self._model.generate_content,
                prompt,
                request_options={"timeout": CONFIG.LLM_TIMEOUT_SECONDS},
            )
            return self._parse_response(memory, response.text)
        except CircuitBreakerOpenError as exc:
            self._logger.error("Gemini CB open in gap analyser", error=str(exc))
            return self._heuristic_fallback(memory)
        except Exception as exc:
            self._logger.error("Gap analyser LLM call failed", error=str(exc))
            return self._heuristic_fallback(memory)

    def _summarise_coverage(self, memory: WorkingMemory) -> str:
        """
        Produce a brief text summary of retrieved chunk coverage.

        Parameters
        ----------
        memory : WorkingMemory

        Returns
        -------
        str
        """
        if not memory.retrieved_chunks:
            return "No chunks retrieved."
        sections = {c.get("metadata", {}).get("section", "unknown") for c in memory.retrieved_chunks}
        fys = {c.get("metadata", {}).get("fiscal_year", "unknown") for c in memory.retrieved_chunks}
        sample_texts = [c.get("text", "")[:200] for c in memory.retrieved_chunks[:3]]
        return (
            f"Chunks retrieved: {len(memory.retrieved_chunks)}\n"
            f"Sections covered: {', '.join(sorted(sections))}\n"
            f"Fiscal years covered: {', '.join(sorted(fys))}\n"
            f"Sample content:\n" + "\n---\n".join(sample_texts)
        )

    def _parse_response(self, memory: WorkingMemory, llm_text: str) -> Tuple[bool, List[str]]:
        """
        Parse LLM JSON response from gap analysis.

        Parameters
        ----------
        memory : WorkingMemory
        llm_text : str

        Returns
        -------
        Tuple[bool, List[str]]
        """
        try:
            clean = re.sub(r"```(?:json)?|```", "", llm_text).strip()
            data = json.loads(clean)
            sufficient = bool(data.get("sufficient", False))
            gaps = data.get("gaps", [])
            if not isinstance(gaps, list):
                gaps = []
            confidence = float(data.get("confidence", 0.5))
            memory.confidence = confidence
            memory.identified_gaps = gaps
            memory.synthesis_ready = sufficient
            return sufficient, gaps
        except (json.JSONDecodeError, ValueError) as exc:
            self._logger.warning("Gap analyser JSON parse failed", error=str(exc))
            return self._heuristic_fallback(memory)

    def _heuristic_fallback(self, memory: WorkingMemory) -> Tuple[bool, List[str]]:
        """
        Simple chunk-count heuristic when LLM is unavailable.

        Parameters
        ----------
        memory : WorkingMemory

        Returns
        -------
        Tuple[bool, List[str]]
        """
        count = len(memory.retrieved_chunks)
        sufficient = count >= 3
        confidence = min(count / 10.0, 0.8) if count > 0 else 0.0
        gaps = [] if sufficient else ["Insufficient chunks retrieved — need more context."]
        memory.confidence = confidence
        memory.identified_gaps = gaps
        memory.synthesis_ready = sufficient
        self._logger.info(
            "Gap analyser heuristic fallback.",
            chunk_count=count,
            sufficient=sufficient,
        )
        return sufficient, gaps


GAP_ANALYSER = GapAnalyser()

# ----------------------------------------------------------------------------
# Cell 18: Gap Analyser
# Purpose: Assess memory coverage and identify missing information.
# Key Classes: GapAnalyser
# Key Functions:
#   GapAnalyser.analyse(memory) → Tuple[bool, List[str]]
#   GapAnalyser._summarise_coverage(memory) → str
#   GapAnalyser._heuristic_fallback(memory) → Tuple[bool, List[str]]
# Key Constants/Config: _GAP_SYSTEM_PROMPT, CONFIG.GEMINI_MODEL
# Imports exported: GapAnalyser, GAP_ANALYSER
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 5 (CB_GEMINI,
#   CircuitBreakerOpenError), Cell 17 (WorkingMemory)
# Critical notes: analyse() mutates memory.confidence, memory.identified_gaps,
#   memory.synthesis_ready as side effects — callers must NOT reset these.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
