"""LangChain chat model factories for the RAG worker."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any

from codebase.ragrun.config import RAGRUN_CONFIG


class LangChainModelFactory:
    """Create LangChain chat models and prompts lazily from project config."""

    def create_mistral(self) -> Any:
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=RAGRUN_CONFIG.mistral_model,
            api_key=RAGRUN_CONFIG.mistral_api_key,
            temperature=RAGRUN_CONFIG.temperature,
            timeout=RAGRUN_CONFIG.timeout_seconds,
            max_tokens=RAGRUN_CONFIG.max_output_tokens,
        )

    def create_gemini(self) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=RAGRUN_CONFIG.gemini_model,
            google_api_key=RAGRUN_CONFIG.gemini_api_key,
            temperature=RAGRUN_CONFIG.temperature,
            max_output_tokens=RAGRUN_CONFIG.max_output_tokens,
            timeout=RAGRUN_CONFIG.timeout_seconds,
        )

    @staticmethod
    def answer_prompt() -> Any:
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = (
            "You are a financial RAG assistant. Answer only from the supplied context. "
            "If the context is insufficient, say you were not able to find anything "
            "that could answer the question."
        )
        human_prompt = (
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Give a concise answer and cite page/source details when available."
        )
        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
