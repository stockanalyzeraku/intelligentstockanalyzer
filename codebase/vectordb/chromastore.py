from __future__ import annotations
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
from typing import Any, Optional
import chromadb

from config import CONFIG
from logger import get_logger
from codebase.vectordb.embedder import EMBEDDER

logger = get_logger(__name__)


class ChromaStore:
    """
    Single access point for ChromaDB operations.
    Singleton — use ChromaStore.get_instance().
    """

    _instance: Optional["ChromaStore"] = None

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, chroma_path: str | None = None):
        self.chroma_path = chroma_path or CONFIG.CHROMA_PATH
        self._client: chromadb.PersistentClient | None = None
        logger.info(f"[ChromaStore] Initialised — path='{self.chroma_path}'")

    # ------------------------------------------------------------------ #
    #  Singleton                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls, chroma_path: str | None = None) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(chroma_path)
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Client                                                              #
    # ------------------------------------------------------------------ #

    def create_client(self) -> chromadb.PersistentClient:
        """Creates or reuses a persistent ChromaDB client."""
        if self._client is None:
            logger.info(f"[ChromaStore] Creating PersistentClient at '{self.chroma_path}'")
            self._client = chromadb.PersistentClient(path=self.chroma_path)
            logger.info("[ChromaStore] Client ready")
        return self._client

    def _get_collection(self, collection_name: str) -> chromadb.Collection:
        """Gets or creates a collection with cosine similarity."""
        client = self.create_client()
        return client.get_or_create_collection(
            name               = collection_name,
            embedding_function = EMBEDDER,
            metadata           = {"hnsw:space": "cosine"},  # set once, can't change later
        )

    # ------------------------------------------------------------------ #
    #  Store                                                               #
    # ------------------------------------------------------------------ #

    def store_in_chromadb(
        self,
        embedding_json_path : str,
        collection_name     : str,
        chroma_path         : str | None = None,
    ) -> chromadb.Collection:
        """
        Loads embedded chunks from a JSON file and upserts into a collection.

        Parameters
        ----------
        embedding_json_path : Path to JSON file with records (id, text, metadata).
        collection_name     : ChromaDB collection name.
        chroma_path         : Optional override for ChromaDB path.

        Returns
        -------
        chromadb.Collection
        """
        if chroma_path:
            self.chroma_path = chroma_path
            self._client     = None

        logger.info(
            f"[ChromaStore] store_in_chromadb — "
            f"collection='{collection_name}', path='{self.chroma_path}'"
        )

        with open(embedding_json_path, "r", encoding="utf-8") as fh:
            records: list[dict] = json.load(fh)
        logger.info(f"[ChromaStore] Loaded {len(records)} chunks")

        ids       : list[str]  = []
        documents : list[str]  = []
        metadatas : list[dict] = []

        for rec in records:
            ids.append(rec["id"])
            documents.append(rec["text"])
            meta = {
                k: v
                for k, v in rec.get("metadata", {}).items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(meta)

        self.upsert_batch(collection_name, ids, documents, metadatas)
        logger.info(f"[ChromaStore] Stored {len(ids)} chunks into '{collection_name}'")
        return self._get_collection(collection_name)

    # ------------------------------------------------------------------ #
    #  Upsert                                                              #
    # ------------------------------------------------------------------ #

    def upsert_batch(
        self,
        collection_name : str,
        ids             : list[str],
        documents       : list[str],
        metadatas       : list[dict],
        batch_size      : int = 100,
    ) -> None:
        """
        Idempotent batch upsert — safe to re-run without duplicate errors.

        Parameters
        ----------
        collection_name : Target collection.
        ids             : Chunk IDs.
        documents       : Text content per chunk.
        metadatas       : Metadata dicts per chunk.
        batch_size      : Number of chunks per upsert call (default 100).
        """
        if not ids:
            logger.warning("[ChromaStore] upsert_batch called with empty ids, skipping.")
            return

        collection = self._get_collection(collection_name)

        for start in range(0, len(ids), batch_size):
            batch_ids  = ids      [start : start + batch_size]
            batch_docs = documents[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            try:
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
                logger.info(f"[ChromaStore] Upserted batch {start}–{start + len(batch_ids)}")
            except Exception as exc:
                logger.error(f"[ChromaStore] Upsert batch failed at {start} — {exc}")
                raise

        logger.info(f"[ChromaStore] Upsert complete — {len(ids)} chunks into '{collection_name}'")

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query_collection(
        self,
        collection_name : str,
        query_texts     : list[str],
        n_results       : int = 10,
        where           : dict | None = None,
    ) -> dict[str, Any]:
        """
        Query a collection by text — ChromaDB embeds the query automatically.

        Parameters
        ----------
        collection_name : Collection to search.
        query_texts     : One or more query strings.
        n_results       : Max results to return.
        where           : Optional metadata filter e.g. {"company": "KALYANKJIL"}.

        Returns
        -------
        dict with keys: ids, documents, metadatas, distances
        """
        collection = self._get_collection(collection_name)
        kwargs: dict[str, Any] = {
            "query_texts" : query_texts,
            "n_results"   : min(n_results, max(collection.count(), 1)),
        }
        if where:
            kwargs["where"] = where

        try:
            return collection.query(**kwargs)
        except Exception as exc:
            logger.error(f"[ChromaStore] Query failed on '{collection_name}' — {exc}")
            raise

    # ------------------------------------------------------------------ #
    #  Get by ID                                                           #
    # ------------------------------------------------------------------ #

    def get_by_id(
        self,
        collection_name : str,
        chunk_id        : str,
    ) -> dict | None:
        """
        Fetch a single chunk by its exact ID.

        Parameters
        ----------
        collection_name : Collection to search.
        chunk_id        : Exact chunk ID.

        Returns
        -------
        dict with keys: id, document, metadata — or None if not found.
        """
        collection = self._get_collection(collection_name)
        try:
            result = collection.get(ids=[chunk_id])
            if result and result["ids"]:
                return {
                    "id"       : result["ids"][0],
                    "document" : result["documents"][0],
                    "metadata" : result["metadatas"][0],
                }
        except Exception as exc:
            logger.error(f"[ChromaStore] get_by_id failed for '{chunk_id}' — {exc}")
        return None

    # ------------------------------------------------------------------ #
    #  Status                                                              #
    # ------------------------------------------------------------------ #

    def status(self, collection_name: str) -> str:
        """
        Returns a summary of chunk count in a collection.

        Parameters
        ----------
        collection_name : Collection to inspect.

        Returns
        -------
        str
        """
        try:
            collection = self._get_collection(collection_name)
            count      = collection.count()
            summary    = f"=== ChromaDB Status ===\n  {collection_name}: {count} chunks"
            logger.info(f"[ChromaStore] {summary}")
            return summary
        except Exception as exc:
            logger.error(f"[ChromaStore] status failed — {exc}")
            return f"Error fetching status: {exc}"