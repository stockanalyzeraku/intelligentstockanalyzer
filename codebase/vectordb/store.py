"""Generic, collection-agnostic record operations against Chroma.

Nothing here knows about parent/child sections — that's ParentChildRetriever's
job (retriever.py), built on top of this. This file also doesn't validate
caller input; that happens once, at the ChromaStore facade boundary, so it
isn't repeated on every internal call.
"""

from __future__ import annotations

from typing import Any

from codebase.vectordb.db import ChromaConnection
from codebase.vectordb.exceptions import QueryError, UpsertError
from codebase.vectordb.schemas import sanitize_metadata, to_records
from codebase.vectordb.skelton import DEFAULT_UPSERT_BATCH_SIZE, ChromaRecord


class ChromaRecordStore:
    """Upsert/query/fetch operations scoped to one ChromaConnection."""

    def __init__(self, connection: ChromaConnection, embedder: Any):
        self._connection = connection
        self._embedder = embedder

    def upsert_records(
        self,
        collection_name: str,
        raw_records: list[dict[str, Any]],
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> int:
        """Upsert a list of {id, text, metadata} dicts. Returns count stored."""
        if not raw_records:
            return 0

        records: list[ChromaRecord] = to_records(raw_records)
        collection = self._connection.get_or_create_collection(collection_name)

        stored = 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            try:
                collection.upsert(
                    ids=[r.id for r in batch],
                    documents=[r.text for r in batch],
                    metadatas=[sanitize_metadata(r.metadata) for r in batch],
                )
            except Exception as exc:
                raise UpsertError(
                    f"Upsert failed for '{collection_name}' at offset {start}"
                ) from exc
            stored += len(batch)
        return stored

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self._connection.get_or_create_collection(collection_name)
        query_embeddings = self._embedder(query_texts)
        try:
            return collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as exc:
            raise QueryError(f"Query failed for '{collection_name}'") from exc

    def get_many_by_ids(self, collection_name: str, chunk_ids: list[str]) -> list[ChromaRecord]:
        """Fetch multiple chunks by exact id, preserving the requested order."""
        if not chunk_ids:
            return []
        collection = self._connection.get_or_create_collection(collection_name)
        result = collection.get(ids=chunk_ids)
        if not result or not result.get("ids"):
            return []
        lookup = {
            cid: ChromaRecord(id=cid, text=doc, metadata=meta or {})
            for cid, doc, meta in zip(
                result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
            )
        }
        return [lookup[cid] for cid in chunk_ids if cid in lookup]

    def get_by_id(self, collection_name: str, chunk_id: str) -> ChromaRecord | None:
        matches = self.get_many_by_ids(collection_name, [chunk_id])
        return matches[0] if matches else None

    def count(self, collection_name: str) -> int:
        return self._connection.get_or_create_collection(collection_name).count()