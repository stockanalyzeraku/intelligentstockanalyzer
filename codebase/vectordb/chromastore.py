"""Public entry point for vector-store operations.

ChromaStore is a thin facade composing three single-purpose pieces:

    db.py        ChromaConnection  — owns the Chroma client + collections
    store.py     ChromaRecordStore — generic upsert/query/fetch
    retriever.py ParentChildRetriever — parent/child-aware search

It's kept as the one import the rest of the app uses (agent tools,
the annual-report agent, health checks, ad-hoc scripts), so splitting
the internals doesn't ripple outward. Input validation happens once,
here at the boundary, via the shared InputValidator — the internal
pieces above stay validation-free and reusable on their own.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from config import CONFIG
from inputvalidator import InputValidator

from codebase.vectordb.db import ChromaConnection
from codebase.vectordb.embedder import EMBEDDER
from codebase.vectordb.retriever import ParentChildRetriever
from codebase.vectordb.schemas import is_bundle_payload, split_bundle
from codebase.vectordb.store import ChromaRecordStore


class ChromaStore:
    """Single access point for ChromaDB operations.

    True singleton: `ChromaStore()` and `ChromaStore.get_instance()` always
    return the same object, so the rest of the app can't end up with two
    independent stores pointed at the same Chroma path.
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
        self._connection = ChromaConnection(chroma_path or CONFIG.CHROMA_DB_PATH, EMBEDDER)
        self._records = ChromaRecordStore(self._connection, EMBEDDER)
        self._retriever = ParentChildRetriever(self._records, CONFIG.COL_PARENT, CONFIG.COL_CHILD)
        self._initialized = True

    @property
    def chroma_path(self) -> str:
        return self._connection.chroma_path

    def use_path(self, chroma_path: str) -> None:
        """Point this store at a different on-disk Chroma path."""
        self._connection.reset_path(chroma_path)

    # ------------------------------------------------------------------ #
    #  Storage                                                            #
    # ------------------------------------------------------------------ #

    def store_in_chromadb(
        self, embedding_json_path: str, collection_name: str, chroma_path: str | None = None
    ) -> int:
        """Load records from a JSON file and upsert them into ChromaDB.
        Returns the number of records stored."""
        collection_name = InputValidator.validate_collection_name(collection_name)
        embedding_json_path = InputValidator.validate_json_path(embedding_json_path, must_exist=True)
        if chroma_path:
            self.use_path(chroma_path)

        with open(embedding_json_path, "r", encoding="utf-8") as fh:
            payload: Any = json.load(fh)
        payload = InputValidator.validate_embedding_payload(payload)

        if is_bundle_payload(payload):
            parents, children = split_bundle(payload)
            stored_parents = self._records.upsert_records(CONFIG.COL_PARENT, parents)
            stored_children = self._records.upsert_records(CONFIG.COL_CHILD, children)
            return stored_parents + stored_children

        return self._records.upsert_records(collection_name, payload)

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
        collection_name = InputValidator.validate_collection_name(collection_name)
        query_texts = InputValidator.validate_query_texts(query_texts)
        where = InputValidator.validate_chroma_where(where)
        n_results = InputValidator.validate_top_k(
            n_results, default=10, max_value=max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K)
        )
        return self._records.query(collection_name, query_texts, n_results, where)

    def query_children_with_parent_context(
        self, query_texts: list[str], n_results: int = 1, where: dict | None = None
    ) -> list[dict[str, Any]]:
        """Search child records first, then return the matching parents."""
        query_texts = InputValidator.validate_query_texts(query_texts)
        where = InputValidator.validate_chroma_where(where)
        n_results = InputValidator.validate_top_k(
            n_results, default=1, max_value=max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K)
        )
        items = self._retriever.retrieve(query_texts, n_results, where)
        return [item.as_dict() for item in items]

    # ------------------------------------------------------------------ #
    #  Lookups / status                                                   #
    # ------------------------------------------------------------------ #

    def get_many_by_ids(self, collection_name: str, chunk_ids: list[str]) -> list[dict]:
        """Fetch multiple chunks by exact id and preserve the requested order."""
        collection_name = InputValidator.validate_collection_name(collection_name)
        records = self._records.get_many_by_ids(collection_name, chunk_ids)
        return [{"id": r.id, "document": r.text, "metadata": r.metadata} for r in records]

    def get_by_id(self, collection_name: str, chunk_id: str) -> dict | None:
        """Fetch a single chunk by its exact ID. Used for debugging."""
        collection_name = InputValidator.validate_collection_name(collection_name)
        record = self._records.get_by_id(collection_name, chunk_id)
        if record is None:
            return None
        return {"id": record.id, "document": record.text, "metadata": record.metadata}

    def collection_count(self, collection_name: str) -> int:
        collection_name = InputValidator.validate_collection_name(collection_name)
        return self._records.count(collection_name)

    def status(self, collection_name: str) -> str:
        """Return a summary of chunk count in a collection."""
        count = self.collection_count(collection_name)
        return f"{collection_name}: {count} chunks"

    @classmethod
    def get_instance(cls, chroma_path: str | None = None) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        return cls(chroma_path)


CHROMASTORE = ChromaStore.get_instance()