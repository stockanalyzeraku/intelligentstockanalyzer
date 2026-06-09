
# =============================================================================
# CELL 16 — Query Understanding Engine
# =============================================================================
"""
Classify, decompose, and expand user queries using the Gemini LLM.
Falls back to safe defaults if LLM call fails.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config import CONFIG
from logger import get_logger
from errorhandler import CB_GEMINI, CircuitBreakerOpenError
from inputvalidator import InputValidator, ValidationError

@dataclass
class QueryIntent:
    """
    Full analysis of one user query.

    Attributes
    ----------
    original_query : str
    category : str
        One of 7 categories.
    sub_queries : List[str]
        Decomposed atomic questions.
    expanded_terms : List[str]
        Synonyms / related phrases for retrieval expansion.
    target_collections : List[str]
        ChromaDB collections to query first.
    scrip_filter : str or None
    fy_filter : str or None
    requires_comparison : bool
    """
    original_query: str
    category: str
    sub_queries: List[str]
    expanded_terms: List[str]
    target_collections: List[str]
    scrip_filter: Optional[str] = None
    fy_filter: Optional[str] = None
    requires_comparison: bool = False


# Maps query category → recommended collections (in priority order)
COLLECTION_MAP: Dict[str, List[str]] = {
    "factual_numerical": [CONFIG.COL_FACTS, CONFIG.COL_CHILD],
    "trend_temporal": [CONFIG.COL_FACTS, CONFIG.COL_CHILD],
    "comparative": [CONFIG.COL_FACTS, CONFIG.COL_CHILD, CONFIG.COL_MGMT],
    "causal": [CONFIG.COL_MGMT, CONFIG.COL_CHILD],
    "strategic": [CONFIG.COL_MGMT, CONFIG.COL_CHILD],
    "risk": [CONFIG.COL_MGMT, CONFIG.COL_CHILD],
    "general": [CONFIG.COL_CHILD, CONFIG.COL_MGMT, CONFIG.COL_FACTS],
}

_VALID_CATEGORIES = set(COLLECTION_MAP.keys())

_QE_SYSTEM_PROMPT = """You are a financial query analysis assistant.
Analyse the given investment research question and return a JSON object with these exact keys:
- category: one of [factual_numerical, trend_temporal, comparative, causal, strategic, risk, general]
- sub_queries: list of 1-4 atomic questions that together answer the original
- expanded_terms: list of 3-8 synonyms/related phrases useful for keyword search
- scrip_filter: company ticker symbol if mentioned, else null
- fy_filter: fiscal year if mentioned (normalise to FY25 format), else null
- requires_comparison: true if the question asks to compare two things, else false

Return ONLY valid JSON. No explanation, no markdown fences."""


class QueryUnderstandingEngine:
    """
    Analyses user queries using the Gemini LLM to produce structured QueryIntent.

    Falls back to _default_intent() if the LLM call fails or returns invalid JSON.
    """

    def __init__(self):
        """Initialise LLM client for query analysis."""
        self._logger = get_logger("query_engine")
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
                    "max_output_tokens": 512,
                },
                system_instruction=_QE_SYSTEM_PROMPT,
            )
        except Exception as exc:
            self._logger.error("Failed to init Gemini for query engine", error=str(exc))
            return None

    def analyse(self, question: str) -> QueryIntent:
        """
        Classify, decompose, and expand a user question.

        Parameters
        ----------
        question : str
            Validated user question.

        Returns
        -------
        QueryIntent
        """
        if self._model is None:
            self._logger.warning("LLM unavailable, using default intent.", question=question[:80])
            return self._default_intent(question)

        try:
            response = CB_GEMINI.call(
                self._model.generate_content,
                question,
                request_options={"timeout": CONFIG.LLM_TIMEOUT_SECONDS},
            )
            return self._parse(question, response.text)
        except CircuitBreakerOpenError as exc:
            self._logger.error("Gemini CB open during query analysis", error=str(exc))
            return self._default_intent(question)
        except Exception as exc:
            self._logger.error("Query analysis LLM call failed", error=str(exc))
            return self._default_intent(question)

    def _parse(self, original_query: str, llm_text: str) -> QueryIntent:
        """
        Parse LLM JSON output into QueryIntent with safe defaults.

        Parameters
        ----------
        original_query : str
        llm_text : str
            Raw LLM response.

        Returns
        -------
        QueryIntent
        """
        try:
            clean = re.sub(r"```(?:json)?|```", "", llm_text).strip()
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            self._logger.warning("JSON parse failed in query engine", error=str(exc))
            return self._default_intent(original_query)

        category = data.get("category", "general")
        if category not in _VALID_CATEGORIES:
            category = "general"

        sub_queries = data.get("sub_queries", [original_query])
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [original_query]

        expanded_terms = data.get("expanded_terms", [])
        if not isinstance(expanded_terms, list):
            expanded_terms = []

        scrip_raw = data.get("scrip_filter")
        scrip_filter = None
        if scrip_raw:
            try:
                scrip_filter = InputValidator.validate_scrip(str(scrip_raw))
            except ValidationError:
                pass

        fy_raw = data.get("fy_filter")
        fy_filter = None
        if fy_raw:
            try:
                fy_filter = InputValidator.validate_fiscal_year(str(fy_raw))
            except ValidationError:
                pass

        return QueryIntent(
            original_query=original_query,
            category=category,
            sub_queries=sub_queries,
            expanded_terms=expanded_terms,
            target_collections=COLLECTION_MAP.get(category, COLLECTION_MAP["general"]),
            scrip_filter=scrip_filter,
            fy_filter=fy_filter,
            requires_comparison=bool(data.get("requires_comparison", False)),
        )

    def _default_intent(self, question: str) -> QueryIntent:
        """
        Return a safe fallback QueryIntent when LLM analysis fails.

        Parameters
        ----------
        question : str

        Returns
        -------
        QueryIntent
        """
        return QueryIntent(
            original_query=question,
            category="general",
            sub_queries=[question],
            expanded_terms=[],
            target_collections=COLLECTION_MAP["general"],
        )


QUERY_ENGINE = QueryUnderstandingEngine()

# ----------------------------------------------------------------------------
# Cell 16: Query Understanding Engine
# Purpose: Classify/decompose/expand queries via LLM into QueryIntent.
# Key Classes: QueryIntent (dataclass), QueryUnderstandingEngine
# Key Functions:
#   QueryUnderstandingEngine.analyse(question) → QueryIntent
#   QueryUnderstandingEngine._parse(original_query, llm_text) → QueryIntent
#   QueryUnderstandingEngine._default_intent(question) → QueryIntent
# Key Constants/Config: COLLECTION_MAP, _VALID_CATEGORIES, _QE_SYSTEM_PROMPT,
#   CONFIG.GEMINI_MODEL, CONFIG.LLM_TIMEOUT_SECONDS
# Imports exported: QueryIntent, QueryUnderstandingEngine, QUERY_ENGINE,
#   COLLECTION_MAP
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 5 (CB_GEMINI,
#   CircuitBreakerOpenError), Cell 6 (InputValidator, ValidationError)
# Critical notes: Always use _default_intent() as fallback — never let LLM
#   failure crash the agent. scrip/fy filters are re-validated via InputValidator.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

