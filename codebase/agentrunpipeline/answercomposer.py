"""Answer composition from retrieved financial context."""

from __future__ import annotations

from typing import Any

from codebase.agentrunpipeline.citationdebugger import Citation
from codebase.agentrunpipeline.models import AnswerGenerator


class FinancialAnswerComposer:
    """Create an answer from retrieved context with visible source markers."""

    def __init__(self, answer_generator: AnswerGenerator | None = None) -> None:
        self.answer_generator = answer_generator

    def compose(self, question: str, context: str, records: list[dict[str, Any]], citations: list[Citation]) -> str:
        """Generate an answer using a custom generator or extractive fallback."""
        if self.answer_generator:
            return self.answer_generator(question, context, records)

        if not citations:
            return "I could not find enough relevant context in the vector store to answer this question."

        evidence_lines = [
            f"{citation.source_id}: {citation.snippet}"
            for citation in citations[:3]
            if citation.snippet
        ]
        return (
            "Based on the retrieved annual-report context, the most relevant evidence is:\n"
            + "\n".join(f"- {line}" for line in evidence_lines)
            + "\n\nUse the generated debug JSON for the full citation metadata and retrieval-tool flow."
        )
