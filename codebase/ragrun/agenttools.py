"""LangChain tools exposed by the RAG worker."""

from __future__ import annotations

from typing import Any

from codebase.ragrun.checkpointer import RuleBasedCheckpointer
from codebase.ragrun.retriever import ChromaRAGRetriever


class RAGAgentTools:
    """Small collection of readable runtime tools for future agent use."""

    def __init__(self, checkpointer: RuleBasedCheckpointer | None = None, retriever: ChromaRAGRetriever | None = None) -> None:
        self.checkpointer = checkpointer or RuleBasedCheckpointer()
        self.retriever = retriever or ChromaRAGRetriever()

    def validate_query(self, query: str) -> dict[str, Any]:
        return self.checkpointer.validate(query).to_dict()

    def search_documents(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.retriever.search(query, top_k=top_k)]

    def as_langchain_tools(self) -> list[Any]:
        """Return LangChain StructuredTool objects without importing LangChain at module import time."""
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(func=self.validate_query, name="validate_query", description="Validate that a financial query includes company, financial year, and topic."),
            StructuredTool.from_function(func=self.search_documents, name="search_documents", description="Search the Chroma annual-report database for relevant chunks."),
        ]
