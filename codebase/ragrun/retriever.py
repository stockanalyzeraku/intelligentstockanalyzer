"""Chroma retrieval for the background RAG worker."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any

from codebase.ragrun.config import RAGRUN_CONFIG
from codebase.ragrun.schemas import RetrievedChunk
from inputvalidator import InputValidator


class ChromaRAGRetriever:
    """Search the existing Chroma database and return readable chunk objects."""

    def __init__(self, chroma_store: Any | None = None) -> None:
        self.chroma_store = chroma_store

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        store = self._store()
        query = InputValidator.validate_question(query)
        limit = InputValidator.validate_top_k(top_k, default=RAGRUN_CONFIG.top_k, max_value=RAGRUN_CONFIG.top_k)
        if hasattr(store, "query_children_with_parent_context"):
            records = store.query_children_with_parent_context([query], n_results=limit)
        else:
            raw = store.query_collection(RAGRUN_CONFIG.collection_name, [query], n_results=limit)
            records = self._raw_to_records(raw)
        return [self._to_chunk(record) for record in records if self._record_text(record)]

    def _store(self) -> Any:
        if self.chroma_store is None:
            from codebase.vectordb.chromastore import ChromaStore

            self.chroma_store = ChromaStore.get_instance(RAGRUN_CONFIG.chroma_path)
        return self.chroma_store

    @staticmethod
    def _raw_to_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        return [
            {"id": cid, "text": doc, "metadata": meta or {}, "distance": dist}
            for cid, doc, meta, dist in zip(ids, docs, metas, dists)
        ]

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        return (
            record.get("child_text")
            or record.get("text")
            or record.get("parent_text")
            or record.get("document")
            or ""
        )

    def _to_chunk(self, record: dict[str, Any]) -> RetrievedChunk:
        metadata = record.get("child_metadata") or record.get("metadata") or record.get("parent_metadata") or {}
        return RetrievedChunk(
            chunk_id=record.get("child_id") or record.get("id"),
            parent_id=record.get("parent_id") or metadata.get("parent_id"),
            text=self._record_text(record),
            page_number=metadata.get("page_number") or metadata.get("page") or metadata.get("page_no"),
            source=metadata.get("source") or metadata.get("file_name") or metadata.get("document_name"),
            company=metadata.get("company"),
            financial_year=metadata.get("report_year") or metadata.get("financial_year") or metadata.get("year"),
            distance=record.get("distance"),
            metadata=metadata,
        )
