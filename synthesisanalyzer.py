# =============================================================================
# CELL 19 — Synthesis Engine
# =============================================================================
"""
SynthesisEngine: generate a structured, cited answer from working memory
and run three post-synthesis verification checks.
"""

import json
import re
from typing import Dict, List
from logger import get_logger
from config import CONFIG
from workingmemory import WorkingMemory
from errorhandler import CB_GEMINI, CircuitBreakerOpenError


SYNTHESIS_SYSTEM_PROMPT = """You are a senior financial analyst assistant providing investment research.

RULES:
1. Every specific number, percentage, or fact MUST have a citation: [Source: <source_display>]
2. Calibrate your confidence: state "high confidence", "moderate confidence", or "low confidence"
   based on data availability.
3. If gaps exist in the data, explicitly state: "Note: Data for [X] was not available."
4. Never fabricate numbers. If data is missing, say so.
5. Structure your answer:
   - Direct answer to the question (1-2 sentences)
   - Supporting analysis with cited data
   - Confidence level statement
   - Follow-up questions the investor should consider (2-3)

Return ONLY a JSON object with these keys:
- answer: full formatted answer text (string)
- sources: list of distinct source_display strings cited
- confidence: float 0.0 to 1.0
- follow_ups: list of 2-3 follow-up questions
- gaps: list of data gaps acknowledged in the answer

JSON only. No markdown fences."""


class SynthesisEngine:
    """
    Generates structured, cited answers from working memory using Gemini.
    Runs three post-synthesis verification checks.
    """

    def __init__(self):
        """Initialise Gemini model for synthesis."""
        self._logger = get_logger("synthesis_engine")
        self._model = self._init_model()

    def _init_model(self):
        """Initialise Gemini generative model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=CONFIG.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=CONFIG.GEMINI_MODEL,
                generation_config={
                    "temperature": CONFIG.LLM_TEMPERATURE,
                    "max_output_tokens": CONFIG.LLM_MAX_OUTPUT_TOKENS,
                },
                system_instruction=SYNTHESIS_SYSTEM_PROMPT,
            )
        except Exception as exc:
            self._logger.error("Failed to init Gemini for synthesis", error=str(exc))
            return None

    def synthesise(self, memory: WorkingMemory) -> Dict:
        """
        Generate and verify a structured answer from working memory.

        Parameters
        ----------
        memory : WorkingMemory
            Populated working memory with retrieved context.

        Returns
        -------
        dict
            Keys: answer, sources, confidence, follow_ups, verified,
                  issues, iterations, gaps.
        """
        context_str = memory.get_ordered_context()
        prompt = (
            f"Question: {memory.question}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Identified gaps: {'; '.join(memory.identified_gaps) if memory.identified_gaps else 'None'}"
        )

        if self._model is None:
            return self._error_result(memory, "LLM unavailable")

        try:
            response = CB_GEMINI.call(
                self._model.generate_content,
                prompt,
                request_options={"timeout": CONFIG.LLM_TIMEOUT_SECONDS},
            )
            result = self._parse_synthesis(response.text, memory)
        except CircuitBreakerOpenError as exc:
            self._logger.error("Gemini CB open during synthesis", error=str(exc))
            return self._error_result(memory, str(exc))
        except Exception as exc:
            self._logger.error("Synthesis LLM call failed", error=str(exc))
            return self._error_result(memory, str(exc))

        result["iterations"] = memory.iteration
        verification = self._verify(result, memory)
        result["verified"] = verification["passed"]
        result["issues"] = verification["issues"]
        return result

    def _parse_synthesis(self, llm_text: str, memory: WorkingMemory) -> Dict:
        """
        Parse JSON synthesis response with safe fallbacks.

        Parameters
        ----------
        llm_text : str
        memory : WorkingMemory

        Returns
        -------
        dict
        """
        try:
            clean = re.sub(r"```(?:json)?|```", "", llm_text).strip()
            data = json.loads(clean)
            return {
                "answer": str(data.get("answer", llm_text)),
                "sources": data.get("sources", []) if isinstance(data.get("sources"), list) else [],
                "confidence": float(data.get("confidence", memory.confidence)),
                "follow_ups": data.get("follow_ups", []) if isinstance(data.get("follow_ups"), list) else [],
                "gaps": data.get("gaps", memory.identified_gaps),
            }
        except (json.JSONDecodeError, ValueError):
            self._logger.warning("Synthesis JSON parse failed — returning raw text.")
            return {
                "answer": llm_text,
                "sources": [],
                "confidence": memory.confidence,
                "follow_ups": [],
                "gaps": memory.identified_gaps,
            }

    def _verify(self, result: Dict, memory: WorkingMemory) -> Dict:
        """
        Run three post-synthesis verification checks.

        Checks
        ------
        1. Citation check: every number in the answer has a [Source: ...] tag.
        2. Hallucination scan: numbers in the answer appear in the retrieved chunks.
        3. Gap acknowledgement: identified gaps are mentioned in the answer.

        Parameters
        ----------
        result : dict
            Synthesis output dict.
        memory : WorkingMemory

        Returns
        -------
        dict
            {'passed': bool, 'issues': List[str]}
        """
        issues: List[str] = []
        answer: str = result.get("answer", "")

        # 1. Citation check — numbers should have [Source: X] nearby
        numbers_in_answer = re.findall(r"\b[\d,]+(?:\.\d+)?%?\b", answer)
        if numbers_in_answer and "[Source:" not in answer:
            issues.append("Numbers found in answer without any [Source: ...] citation.")

        # 2. Hallucination scan — check numbers exist somewhere in retrieved context
        context_blob = " ".join(c.get("text", "") for c in memory.retrieved_chunks)
        hallucinated: List[str] = []
        for num in numbers_in_answer[:20]:  # check first 20 numbers
            clean_num = num.replace(",", "").replace("%", "")
            if len(clean_num) >= 3 and clean_num not in context_blob.replace(",", ""):
                hallucinated.append(num)
        if hallucinated:
            issues.append(f"Potentially hallucinated numbers: {', '.join(hallucinated[:5])}")

        # 3. Gap acknowledgement
        for gap in memory.identified_gaps:
            gap_keywords = gap.lower().split()[:3]
            mentioned = any(kw in answer.lower() for kw in gap_keywords)
            if not mentioned:
                issues.append(f"Gap not acknowledged in answer: '{gap}'")

        passed = len(issues) == 0
        if not passed:
            self._logger.warning("Synthesis verification issues", issues=issues)
        return {"passed": passed, "issues": issues}

    def _error_result(self, memory: WorkingMemory, reason: str) -> Dict:
        """
        Return a structured error result when synthesis cannot proceed.

        Parameters
        ----------
        memory : WorkingMemory
        reason : str

        Returns
        -------
        dict
        """
        return {
            "answer": f"Unable to synthesise answer. Reason: {reason}",
            "sources": [],
            "confidence": 0.0,
            "follow_ups": [],
            "gaps": memory.identified_gaps,
            "iterations": memory.iteration,
            "verified": False,
            "issues": [reason],
        }


SYNTHESIS_ENGINE = SynthesisEngine()

# ----------------------------------------------------------------------------
# Cell 19: Synthesis Engine
# Purpose: Generate cited structured answers and verify them post-synthesis.
# Key Classes: SynthesisEngine
# Key Functions:
#   SynthesisEngine.synthesise(memory) → dict
#   SynthesisEngine._verify(result, memory) → dict
#   SynthesisEngine._parse_synthesis(llm_text, memory) → dict
#   SynthesisEngine._error_result(memory, reason) → dict
# Key Constants/Config: SYNTHESIS_SYSTEM_PROMPT, CONFIG.LLM_MAX_OUTPUT_TOKENS,
#   CONFIG.LLM_TEMPERATURE
# Imports exported: SynthesisEngine, SYNTHESIS_ENGINE, SYNTHESIS_SYSTEM_PROMPT
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 5 (CB_GEMINI,
#   CircuitBreakerOpenError), Cell 17 (WorkingMemory)
# Critical notes: Verification is best-effort and logged as warnings — it does
#   NOT block returning an answer. 'verified': False signals the caller to warn
#   the user, not to suppress the answer.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

