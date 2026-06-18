"""Answer composition from retrieved financial context."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, Iterable

from codebase.agentrunpipeline.citationdebugger import Citation
from codebase.agentrunpipeline.models import AnswerGenerator, AnswerProvider, LLMModelConfig


class LangChainAnswerProvider:
    """Answer provider backed by a LangChain chat model."""

    def __init__(self, config: LLMModelConfig) -> None:
        self.config = config
        self.name = f"{config.provider}:{config.model}"
        self._chat_model: Any | None = None

    def generate(self, question: str, context: str, records: list[dict[str, Any]]) -> str:
        """Generate an answer using the configured chat model."""
        messages = [
            ("system", self._system_prompt()),
            ("human", self._user_prompt(question, context, records)),
        ]
        response = self._get_chat_model().invoke(messages)
        return self._stringify_response(response)

    def _get_chat_model(self) -> Any:
        if self._chat_model is not None:
            return self._chat_model

        provider = self.config.provider.lower()
        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI

            self._chat_model = ChatMistralAI(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                timeout=self.config.timeout_seconds,
                max_tokens=self.config.max_output_tokens,
            )
        elif provider in {"google", "gemini", "google_genai"}:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._chat_model = ChatGoogleGenerativeAI(
                model=self.config.model,
                google_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                timeout=self.config.timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported answer provider: {self.config.provider}")
        return self._chat_model

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You answer questions about annual-report and financial-report context. "
            "Use only the supplied context. If the context is insufficient, say so. "
            "Cite source markers exactly as they appear in the context, such as [Source 1]."
        )

    @staticmethod
    def _user_prompt(question: str, context: str, records: list[dict[str, Any]]) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Retrieved record count: {len(records)}\n"
            "Write a concise, grounded answer with source markers."
        )

    @staticmethod
    def _stringify_response(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return " ".join(parts)
        return str(content)


class FallbackAnswerGenerator:
    """Try multiple answer providers in order until one returns text."""

    def __init__(self, providers: Iterable[AnswerProvider]) -> None:
        self.providers = list(providers)

    def __call__(self, question: str, context: str, records: list[dict[str, Any]]) -> str:
        errors: list[str] = []
        for provider in self.providers:
            try:
                answer = provider.generate(question, context, records).strip()
                if answer:
                    return answer
                errors.append(f"{provider.name}: empty response")
            except Exception as exc:  # providers are isolated so fallbacks can run
                errors.append(f"{provider.name}: {exc}")
        raise RuntimeError("All answer providers failed: " + "; ".join(errors))


def build_default_answer_generator() -> AnswerGenerator:
    """Build the default Mistral-first, Gemini-fallback answer generator."""
    from config import CONFIG

    configs = [
        LLMModelConfig(
            provider="mistral",
            model=getattr(CONFIG, "MISTRAL_ANSWER_MODEL", "open-mistral-nemo"),
            api_key=CONFIG.MISTRAL_API_KEY,
            temperature=0.0,
            max_output_tokens=CONFIG.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=CONFIG.LLM_TIMEOUT_SECONDS,
        ),
        LLMModelConfig(
            provider="google",
            model=CONFIG.GEMINI_MODEL,
            api_key=CONFIG.GEMINI_API_KEY,
            temperature=0.0,
            max_output_tokens=CONFIG.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=CONFIG.LLM_TIMEOUT_SECONDS,
        ),
    ]
    return FallbackAnswerGenerator(LangChainAnswerProvider(config) for config in configs)


class FinancialAnswerComposer:
    """Create an answer from retrieved context with visible source markers."""

    def __init__(
        self,
        answer_generator: AnswerGenerator | None = None,
        *,
        use_default_llm: bool = False,
    ) -> None:
        self.answer_generator = answer_generator or (build_default_answer_generator() if use_default_llm else None)

    def compose(self, question: str, context: str, records: list[dict[str, Any]], citations: list[Citation]) -> str:
        """Generate an answer using configured LLMs or an extractive fallback."""
        if not citations:
            return "I could not find enough relevant context in the vector store to answer this question."

        if self.answer_generator:
            try:
                return self.answer_generator(question, context, records)
            except Exception as exc:
                return self._extractive_answer(citations, provider_error=str(exc))

        return self._extractive_answer(citations)

    @staticmethod
    def _extractive_answer(citations: list[Citation], provider_error: str | None = None) -> str:
        evidence_lines = [
            f"{citation.source_id}: {citation.snippet}"
            for citation in citations[:3]
            if citation.snippet
        ]
        prefix = "Based on the retrieved annual-report context, the most relevant evidence is:\n"
        suffix = "\n\nUse the generated debug JSON for the full citation metadata and retrieval-tool flow."
        if provider_error:
            suffix += f"\n\nLLM answer generation failed, so this extractive fallback was used: {provider_error}"
        return prefix + "\n".join(f"- {line}" for line in evidence_lines) + suffix
