"""Retrieval tools that wrap the existing ChromaStore API."""

from __future__ import annotations

from typing import Any

from codebase.agentrunpipeline.citationdebugger import ToolTrace


class FinancialRetrievalTools:
    """Retrieve child chunks with parent-page context from ChromaStore."""

    def __init__(self, chroma_store: Any | None = None) -> None:
        self._chroma_store = chroma_store

    @property
    def chroma_store(self) -> Any:
        """Lazily import ChromaStore so simple tests do not need Chroma dependencies."""
        if self._chroma_store is None:
            from codebase.vectordb.chromastore import ChromaStore

            self._chroma_store = ChromaStore.get_instance()
        return self._chroma_store

    def child_parent_search(
        self,
        queries: list[str],
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> tuple[list[dict[str, Any]], ToolTrace]:
        """Search child chunks and return parent-expanded records."""
        results = self.chroma_store.query_children_with_parent_context(
            query_texts=queries,
            n_results=top_k,
            where=filters or None,
        )
        trace = ToolTrace(
            tool_name="child_parent_search",
            input={"queries": queries, "filters": filters or {}, "top_k": top_k},
            output_summary={"result_count": len(results)},
        )
        return results, trace
