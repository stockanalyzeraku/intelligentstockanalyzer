"""Public entry point for vector-store operations.

ChromaStore is a thin facade composing three single-purpose pieces:

    db.py        ChromaConnection       — owns the Chroma client + collections
    store.py     ChromaRecordStore      — generic upsert/query/fetch
    retriever.py ParentChildRetriever   — parent/child-aware search

The logger is created here (the entry point, equivalent to runner.py in
fileloader) and passed down through every internal call so the full
trace — validation, client creation, batching, retrieval — lives in one
log stream per identifier.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from config import CONFIG
from logger import StructuredLogger, get_logger

from codebase.vectordb.db import ChromaConnection
from codebase.vectordb.embedder import EMBEDDER
from codebase.vectordb.exceptions import (
    InvalidCollectionNameError,
    InvalidIdError,
    InvalidPathError,
    InvalidPayloadShapeError,
    InvalidQueryTextError,
    InvalidTopKError,
    UnsafeFilterError,
    VectorDBValidationError,
)
from codebase.vectordb.retriever import ParentChildRetriever
from codebase.vectordb.schemas import is_bundle_payload, split_bundle
from codebase.vectordb.store import ChromaRecordStore
from codebase.vectordb.validator import (
    validate_chunk_id,
    validate_chunk_ids,
    validate_chroma_path,
    validate_collection_name,
    validate_embedding_payload,
    validate_json_path,
    validate_query_texts,
    validate_top_k,
    validate_where_filter,
)


class ChromaStore:
    """Single access point for ChromaDB operations.

    True singleton: ChromaStore() and ChromaStore.get_instance() always
    return the same object.
    """

    _instance: Optional["ChromaStore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, chroma_path: str | None = None):
        if getattr(self, "_initialized", False):
            return
        resolved_path = chroma_path or CONFIG.CHROMA_DB_PATH
        self._logger: StructuredLogger = get_logger("VECTORDB", resolved_path)
        self._connection = ChromaConnection(resolved_path, EMBEDDER)
        self._records    = ChromaRecordStore(self._connection, EMBEDDER)
        self._retriever  = ParentChildRetriever(
            self._records, CONFIG.COL_PARENT, CONFIG.COL_CHILD
        )
        self._initialized = True
        self._logger.event(
            f"{resolved_path} : ChromaStore initialised",
            step="init", outcome="passed", chroma_path=resolved_path,
        )

    @property
    def chroma_path(self) -> str:
        return self._connection.chroma_path

    def use_path(self, chroma_path: str) -> None:
        """Point this store at a different on-disk Chroma path."""
        try:
            validated = validate_chroma_path(chroma_path)
        except InvalidPathError as exc:
            self._logger.event(
                f"{chroma_path} : Chroma path validation failed: {exc}",
                step="use_path", outcome="failed", chroma_path=chroma_path,
            )
            raise
        self._connection.reset_path(validated)
        self._logger.event(
            f"{validated} : Chroma path updated",
            step="use_path", outcome="passed", chroma_path=validated,
        )

    # ------------------------------------------------------------------ #
    #  Storage                                                            #
    # ------------------------------------------------------------------ #

    def store_in_chromadb(
        self,
        embedding_json_path: str,
        collection_name: str,
        chroma_path: str | None = None,
    ) -> int:
        """Load records from a JSON file and upsert them into ChromaDB.
        Returns the total number of records stored."""

        self._logger.event(
            f"{collection_name} : store_in_chromadb started",
            step="store_in_chromadb", stage="start",
            collection=collection_name, json_path=embedding_json_path,
        )

        # -- Validate all external inputs first -------------------------
        try:
            collection_name = validate_collection_name(collection_name)
        except InvalidCollectionNameError as exc:
            self._logger.event(
                f"{collection_name} : Collection name validation failed: {exc}",
                step="store_in_chromadb", outcome="failed",
                field="collection_name", exception_type=type(exc).__name__,
            )
            raise

        self._logger.event(
            f"{collection_name} : Collection name validated",
            step="validate_collection_name", outcome="passed",
            collection=collection_name,
        )

        try:
            embedding_json_path = validate_json_path(
                embedding_json_path, base_dir=CONFIG.UPLOADS_PATH
            )
        except InvalidPathError as exc:
            self._logger.event(
                f"{collection_name} : JSON path validation failed: {exc}",
                step="store_in_chromadb", outcome="failed",
                field="embedding_json_path", exception_type=type(exc).__name__,
            )
            raise

        self._logger.event(
            f"{collection_name} : JSON path validated — {embedding_json_path}",
            step="validate_json_path", outcome="passed",
            json_path=embedding_json_path,
        )

        if chroma_path:
            try:
                self.use_path(chroma_path)
            except InvalidPathError:
                raise

        # -- Load and validate JSON payload -----------------------------
        self._logger.event(
            f"{collection_name} : Loading JSON payload from {embedding_json_path}",
            step="load_json", stage="start", json_path=embedding_json_path,
        )

        with open(embedding_json_path, "r", encoding="utf-8") as fh:
            payload: Any = json.load(fh)

        self._logger.event(
            f"{collection_name} : JSON loaded",
            step="load_json", outcome="passed", json_path=embedding_json_path,
        )

        try:
            payload = validate_embedding_payload(payload)
        except (InvalidPayloadShapeError, VectorDBValidationError) as exc:
            self._logger.event(
                f"{collection_name} : Payload validation failed: {exc}",
                step="validate_payload", outcome="failed",
                exception_type=type(exc).__name__,
            )
            raise

        # -- Route bundle vs flat list and upsert -----------------------
        if is_bundle_payload(payload):
            parents, children = split_bundle(payload)
            self._logger.event(
                f"{collection_name} : Bundle payload detected — "
                f"{len(parents)} parent(s), {len(children)} child(ren)",
                step="validate_payload", outcome="passed",
                payload_type="bundle", parents=len(parents), children=len(children),
            )
            stored_parents  = self._records.upsert_records(
                CONFIG.COL_PARENT, parents, self._logger
            )
            stored_children = self._records.upsert_records(
                CONFIG.COL_CHILD, children, self._logger
            )
            total = stored_parents + stored_children
        else:
            self._logger.event(
                f"{collection_name} : Flat record list detected — "
                f"{len(payload)} record(s)",
                step="validate_payload", outcome="passed",
                payload_type="flat", records=len(payload),
            )
            total = self._records.upsert_records(collection_name, payload, self._logger)

        self._logger.event(
            f"{collection_name} : store_in_chromadb complete — {total} record(s) stored",
            step="store_in_chromadb", outcome="passed",
            collection=collection_name, stored=total,
        )
        return total

    # ------------------------------------------------------------------ #
    #  Query                                                              #
    # ------------------------------------------------------------------ #

    def query_collection(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:

        self._logger.event(
            f"{collection_name} : query_collection started",
            step="query_collection", stage="start", collection=collection_name,
        )

        try:
            collection_name = validate_collection_name(collection_name)
        except InvalidCollectionNameError as exc:
            self._logger.event(
                f"{collection_name} : Collection name validation failed: {exc}",
                step="query_collection", outcome="failed",
                field="collection_name", exception_type=type(exc).__name__,
            )
            raise

        try:
            query_texts = validate_query_texts(query_texts)
        except InvalidQueryTextError as exc:
            self._logger.event(
                f"{collection_name} : Query text validation failed: {exc}",
                step="query_collection", outcome="failed",
                field="query_texts", exception_type=type(exc).__name__,
            )
            raise

        try:
            n_results = validate_top_k(
                n_results,
                default=10,
                max_value=max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K),
            )
        except InvalidTopKError as exc:
            self._logger.event(
                f"{collection_name} : top_k validation failed: {exc}",
                step="query_collection", outcome="failed",
                field="n_results", exception_type=type(exc).__name__,
            )
            raise

        try:
            where = validate_where_filter(where)
        except UnsafeFilterError as exc:
            self._logger.event(
                f"{collection_name} : where filter validation failed: {exc}",
                step="query_collection", outcome="failed",
                field="where", exception_type=type(exc).__name__,
            )
            raise

        self._logger.event(
            f"{collection_name} : All inputs validated",
            step="query_collection", stage="inputs_validated",
            collection=collection_name, n_results=n_results,
        )

        result = self._records.query(
            collection_name, query_texts, n_results, self._logger, where
        )

        self._logger.event(
            f"{collection_name} : query_collection complete",
            step="query_collection", outcome="passed", collection=collection_name,
        )
        return result

    def query_children_with_parent_context(
        self,
        query_texts: list[str],
        n_results: int = 1,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search child chunks first, then return the matching parents."""

        self._logger.event(
            "query_children_with_parent_context started",
            step="query_children_with_parent_context", stage="start",
        )

        try:
            query_texts = validate_query_texts(query_texts)
        except InvalidQueryTextError as exc:
            self._logger.event(
                f"Query text validation failed: {exc}",
                step="query_children_with_parent_context", outcome="failed",
                field="query_texts", exception_type=type(exc).__name__,
            )
            raise

        try:
            n_results = validate_top_k(
                n_results,
                default=1,
                max_value=max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K),
            )
        except InvalidTopKError as exc:
            self._logger.event(
                f"top_k validation failed: {exc}",
                step="query_children_with_parent_context", outcome="failed",
                field="n_results", exception_type=type(exc).__name__,
            )
            raise

        try:
            where = validate_where_filter(where)
        except UnsafeFilterError as exc:
            self._logger.event(
                f"where filter validation failed: {exc}",
                step="query_children_with_parent_context", outcome="failed",
                field="where", exception_type=type(exc).__name__,
            )
            raise

        self._logger.event(
            "All inputs validated",
            step="query_children_with_parent_context", stage="inputs_validated",
            n_results=n_results,
        )

        items = self._retriever.retrieve(query_texts, n_results, self._logger, where)
        result = [item.as_dict() for item in items]

        self._logger.event(
            f"query_children_with_parent_context complete — {len(result)} item(s)",
            step="query_children_with_parent_context", outcome="passed",
            returned=len(result),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Lookups / status                                                   #
    # ------------------------------------------------------------------ #

    def get_many_by_ids(self, collection_name: str, chunk_ids: list[str]) -> list[dict]:
        """Fetch multiple chunks by exact id, preserving the requested order."""

        self._logger.event(
            f"{collection_name} : get_many_by_ids started — {len(chunk_ids)} id(s)",
            step="get_many_by_ids", stage="start",
            collection=collection_name, requested=len(chunk_ids),
        )

        try:
            collection_name = validate_collection_name(collection_name)
        except InvalidCollectionNameError as exc:
            self._logger.event(
                f"{collection_name} : Collection name validation failed: {exc}",
                step="get_many_by_ids", outcome="failed",
                field="collection_name", exception_type=type(exc).__name__,
            )
            raise

        try:
            chunk_ids = validate_chunk_ids(chunk_ids)
        except InvalidIdError as exc:
            self._logger.event(
                f"{collection_name} : chunk_ids validation failed: {exc}",
                step="get_many_by_ids", outcome="failed",
                field="chunk_ids", exception_type=type(exc).__name__,
            )
            raise

        records = self._records.get_many_by_ids(collection_name, chunk_ids, self._logger)
        result  = [{"id": r.id, "document": r.text, "metadata": r.metadata} for r in records]

        self._logger.event(
            f"{collection_name} : get_many_by_ids complete — "
            f"{len(result)}/{len(chunk_ids)} found",
            step="get_many_by_ids", outcome="passed",
            collection=collection_name, requested=len(chunk_ids), found=len(result),
        )
        return result

    def get_by_id(self, collection_name: str, chunk_id: str) -> dict | None:
        """Fetch a single chunk by its exact ID. Used for debugging."""

        self._logger.event(
            f"{collection_name} : get_by_id started — '{chunk_id}'",
            step="get_by_id", stage="start",
            collection=collection_name, chunk_id=chunk_id,
        )

        try:
            collection_name = validate_collection_name(collection_name)
        except InvalidCollectionNameError as exc:
            self._logger.event(
                f"{collection_name} : Collection name validation failed: {exc}",
                step="get_by_id", outcome="failed",
                field="collection_name", exception_type=type(exc).__name__,
            )
            raise

        try:
            chunk_id = validate_chunk_id(chunk_id)
        except InvalidIdError as exc:
            self._logger.event(
                f"{collection_name} : chunk_id validation failed: {exc}",
                step="get_by_id", outcome="failed",
                field="chunk_id", exception_type=type(exc).__name__,
            )
            raise

        record = self._records.get_by_id(collection_name, chunk_id, self._logger)
        if record is None:
            self._logger.event(
                f"{collection_name} : chunk '{chunk_id}' not found",
                step="get_by_id", outcome="passed",
                collection=collection_name, chunk_id=chunk_id, found=False,
            )
            return None

        self._logger.event(
            f"{collection_name} : chunk '{chunk_id}' found",
            step="get_by_id", outcome="passed",
            collection=collection_name, chunk_id=chunk_id, found=True,
        )
        return {"id": record.id, "document": record.text, "metadata": record.metadata}

    def collection_count(self, collection_name: str) -> int:
        try:
            collection_name = validate_collection_name(collection_name)
        except InvalidCollectionNameError as exc:
            self._logger.event(
                f"{collection_name} : Collection name validation failed: {exc}",
                step="collection_count", outcome="failed",
                field="collection_name", exception_type=type(exc).__name__,
            )
            raise
        return self._records.count(collection_name, self._logger)

    def status(self, collection_name: str) -> str:
        """Return a human-readable chunk count for a collection."""
        count = self.collection_count(collection_name)
        summary = f"{collection_name}: {count} chunks"
        self._logger.event(
            f"{collection_name} : Status — {summary}",
            step="status", outcome="passed",
            collection=collection_name, count=count,
        )
        return summary

    @classmethod
    def get_instance(cls, chroma_path: str | None = None) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        return cls(chroma_path)


CHROMASTORE = ChromaStore.get_instance()