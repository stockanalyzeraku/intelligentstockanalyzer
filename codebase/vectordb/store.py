"""Generic, collection-agnostic record operations against Chroma.

Nothing here knows about parent/child sections — that is ParentChildRetriever's
job (retriever.py), built on top of this.
"""

from __future__ import annotations

from typing import Any

from logger import StructuredLogger

from codebase.vectordb.db import ChromaConnection
from codebase.vectordb.exceptions import QueryError, UpsertError
from codebase.vectordb.schemas import sanitize_metadata, to_records
from codebase.vectordb.skelton import DEFAULT_UPSERT_BATCH_SIZE, ChromaRecord
from codebase.vectordb.validator import (
    validate_batch_size,
    validate_embedding_vectors,
    validate_records,
)
from config import CONFIG


class ChromaRecordStore:
    """Upsert/query/fetch operations scoped to one ChromaConnection."""

    def __init__(self, connection: ChromaConnection, embedder: Any):
        self._connection = connection
        self._embedder = embedder

    def upsert_records(
        self,
        collection_name: str,
        raw_records: list[dict[str, Any]],
        logger: StructuredLogger,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> int:
        """Validate and upsert a list of {id, text, metadata} dicts.
        Returns the total number of records stored."""

        logger.event(
            f"{collection_name} : Upsert started — {len(raw_records)} record(s)",
            step="upsert_records", stage="start",
            collection=collection_name, record_count=len(raw_records),
        )

        # Validate every record going into the database (defense in depth —
        # chromastore.py has already validated the payload shape; this catches
        # anything that bypasses the facade and calls store directly).
        try:
            validated_records = validate_records(raw_records)
            validated_batch_size = validate_batch_size(batch_size)
        except Exception as exc:
            logger.event(
                f"{collection_name} : Record validation failed before upsert: {exc}",
                step="upsert_records", outcome="failed",
                collection=collection_name, exception_type=type(exc).__name__,
            )
            raise

        logger.event(
            f"{collection_name} : All {len(validated_records)} record(s) validated",
            step="upsert_records", outcome="passed",
            collection=collection_name, record_count=len(validated_records),
        )

        records = to_records(validated_records)
        collection = self._connection.get_or_create_collection(collection_name, logger)

        stored = 0
        total_batches = (len(records) + validated_batch_size - 1) // validated_batch_size

        for batch_index, start in enumerate(range(0, len(records), validated_batch_size), 1):
            batch = records[start : start + validated_batch_size]
            logger.event(
                f"{collection_name} : Upserting batch {batch_index}/{total_batches} "
                f"({len(batch)} record(s))",
                step="upsert_batch", stage="start",
                collection=collection_name, batch=batch_index,
                batch_size=len(batch), offset=start,
            )
            try:
                collection.upsert(
                    ids=[r.id for r in batch],
                    documents=[r.text for r in batch],
                    metadatas=[sanitize_metadata(r.metadata) for r in batch],
                )
            except Exception as exc:
                logger.error(
                    f"{collection_name} : Batch {batch_index} upsert failed: {exc}",
                    step="upsert_batch", outcome="failed",
                    collection=collection_name, batch=batch_index,
                    offset=start, exception_type=type(exc).__name__,
                )
                raise UpsertError(
                    f"Upsert failed for '{collection_name}' at batch {batch_index} (offset {start})"
                ) from exc

            stored += len(batch)
            logger.event(
                f"{collection_name} : Batch {batch_index}/{total_batches} upserted "
                f"({stored}/{len(records)} total)",
                step="upsert_batch", outcome="passed",
                collection=collection_name, batch=batch_index, stored_so_far=stored,
            )

        logger.event(
            f"{collection_name} : Upsert complete — {stored} record(s) stored",
            step="upsert_records", outcome="passed",
            collection=collection_name, stored=stored,
        )
        return stored

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int,
        logger: StructuredLogger,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        logger.event(
            f"{collection_name} : Query started — {len(query_texts)} text(s), "
            f"top_k={n_results}",
            step="query", stage="start",
            collection=collection_name, query_count=len(query_texts), n_results=n_results,
        )

        query_embeddings = self._embedder(query_texts)

        # Validate embedding vectors before they reach Chroma
        try:
            validate_embedding_vectors(query_embeddings, expected_dim=CONFIG.EMBEDDING_DIM)
        except Exception as exc:
            logger.event(
                f"{collection_name} : Embedding validation failed: {exc}",
                step="query", outcome="failed",
                collection=collection_name, exception_type=type(exc).__name__,
            )
            raise

        collection = self._connection.get_or_create_collection(collection_name, logger)
        try:
            result = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as exc:
            logger.error(
                f"{collection_name} : Query failed: {exc}",
                step="query", outcome="failed",
                collection=collection_name, exception_type=type(exc).__name__,
            )
            raise QueryError(f"Query failed for '{collection_name}'") from exc

        returned = len(result.get("ids", [[]])[0]) if result.get("ids") else 0
        logger.event(
            f"{collection_name} : Query complete — {returned} result(s) returned",
            step="query", outcome="passed",
            collection=collection_name, returned=returned,
        )
        return result

    def get_many_by_ids(
        self, collection_name: str, chunk_ids: list[str], logger: StructuredLogger
    ) -> list[ChromaRecord]:
        """Fetch multiple chunks by exact id, preserving the requested order."""
        if not chunk_ids:
            return []

        logger.event(
            f"{collection_name} : Fetching {len(chunk_ids)} chunk(s) by id",
            step="get_many_by_ids", stage="start",
            collection=collection_name, requested=len(chunk_ids),
        )

        collection = self._connection.get_or_create_collection(collection_name, logger)
        result = collection.get(ids=chunk_ids)

        if not result or not result.get("ids"):
            logger.event(
                f"{collection_name} : No chunks found for requested ids",
                step="get_many_by_ids", outcome="passed",
                collection=collection_name, requested=len(chunk_ids), found=0,
            )
            return []

        lookup = {
            cid: ChromaRecord(id=cid, text=doc, metadata=meta or {})
            for cid, doc, meta in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        }
        records = [lookup[cid] for cid in chunk_ids if cid in lookup]

        logger.event(
            f"{collection_name} : Fetched {len(records)}/{len(chunk_ids)} chunk(s)",
            step="get_many_by_ids", outcome="passed",
            collection=collection_name, requested=len(chunk_ids), found=len(records),
        )
        return records

    def get_by_id(
        self, collection_name: str, chunk_id: str, logger: StructuredLogger
    ) -> ChromaRecord | None:
        logger.event(
            f"{collection_name} : Fetching single chunk '{chunk_id}'",
            step="get_by_id", stage="start",
            collection=collection_name, chunk_id=chunk_id,
        )
        matches = self.get_many_by_ids(collection_name, [chunk_id], logger)
        found = matches[0] if matches else None
        logger.event(
            f"{collection_name} : Single chunk '{chunk_id}' "
            f"{'found' if found else 'not found'}",
            step="get_by_id", outcome="passed",
            collection=collection_name, chunk_id=chunk_id, found=found is not None,
        )
        return found

    def count(self, collection_name: str, logger: StructuredLogger) -> int:
        return self._connection.get_or_create_collection(collection_name, logger).count()