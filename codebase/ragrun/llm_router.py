"""Mistral-first, Gemini-fallback answer generation."""

from __future__ import annotations

from typing import Any

from codebase.ragrun.models import LangChainModelFactory
from codebase.ragrun.schemas import ModelRunResult, RetrievedChunk


class LLMRouter:
    """Ask Mistral first and fall back to Gemini when needed."""

    INSUFFICIENT_MARKERS = (
        "cannot answer", "can't answer", "insufficient context", "not enough context",
        "unable to answer", "do not have enough", "not able to find",
    )

    def __init__(self, model_factory: LangChainModelFactory | None = None) -> None:
        self.model_factory = model_factory or LangChainModelFactory()

    def answer(self, query: str, chunks: list[RetrievedChunk]) -> ModelRunResult:
        context = self._format_context(chunks)
        attempts: list[dict[str, Any]] = []
        providers = [
            ("mistral", "mistral-large-latest", self.model_factory.create_mistral),
            ("gemini", "gemini", self.model_factory.create_gemini),
        ]

        for provider, model_name, factory in providers:
            try:
                answer = self._invoke(factory(), query, context).strip()
                good_enough = bool(answer) and not self._is_insufficient(answer)
                attempts.append(
                    {
                        "provider": provider,
                        "model": model_name,
                        "status": "answered" if good_enough else "insufficient",
                        "answer_preview": answer[:240],
                    }
                )
                if good_enough:
                    return ModelRunResult(
                        answer=answer,
                        model_used=model_name,
                        provider_used=provider,
                        fallback_used=provider != "mistral",
                        attempts=attempts,
                    )
            except Exception as exc:
                attempts.append(
                    {
                        "provider": provider,
                        "model": model_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        return ModelRunResult(
            answer="I was not able to generate an answer from the retrieved context.",
            model_used=None,
            provider_used=None,
            fallback_used=True,
            attempts=attempts,
        )

    def _invoke(self, model: Any, query: str, context: str) -> str:
        prompt = self.model_factory.answer_prompt()
        chain = prompt | model
        response = chain.invoke({"question": query, "context": context})
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            location = f"source={chunk.source or 'unknown'}, page={chunk.page_number or 'unknown'}"
            blocks.append(f"[Chunk {index}; {location}]\n{chunk.text}")
        return "\n\n".join(blocks)

    def _is_insufficient(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(marker in lowered for marker in self.INSUFFICIENT_MARKERS)
