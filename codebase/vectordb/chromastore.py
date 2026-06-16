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
        self._logger = logger
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

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Keep only scalar metadata values that ChromaDB can store."""
        return {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

    def _upsert_records(self, collection_name: str, records: list[dict]) -> None:
        """Upsert a list of records into one collection."""
        if not records:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for rec in records:
            ids.append(rec["id"])
            documents.append(rec["text"])
            metadatas.append(self._sanitize_metadata(rec.get("metadata", {})))

        self.upsert_batch(collection_name, ids, documents, metadatas)

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
        Loads records from a JSON file and upserts them into ChromaDB.

        Supports:
        - legacy flat record lists, stored into ``collection_name``
        - parent/child bundles with ``{"parents": [...], "children": [...]}``
          stored into ``CONFIG.COL_PARENT`` and ``CONFIG.COL_CHILD``

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
            payload: Any = json.load(fh)

        if isinstance(payload, dict) and "parents" in payload and "children" in payload:
            parents = payload.get("parents", [])
            children = payload.get("children", [])
            logger.info(
                f"[ChromaStore] Loaded parent/child bundle — parents={len(parents)}, children={len(children)}"
            )
            self._upsert_records(CONFIG.COL_PARENT, parents)
            self._upsert_records(CONFIG.COL_CHILD, children)
            logger.info(
                f"[ChromaStore] Stored bundle into '{CONFIG.COL_PARENT}' and '{CONFIG.COL_CHILD}'"
            )
            return self._get_collection(CONFIG.COL_CHILD)

        records: list[dict] = payload
        logger.info(f"[ChromaStore] Loaded {len(records)} legacy records")
        self._upsert_records(collection_name, records)
        logger.info(f"[ChromaStore] Stored {len(records)} records into '{collection_name}'")
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

    def get_many_by_ids(
        self,
        collection_name: str,
        chunk_ids: list[str],
    ) -> list[dict]:
        """Fetch multiple chunks by exact id and preserve the requested order."""
        if not chunk_ids:
            return []

        collection = self._get_collection(collection_name)
        try:
            result = collection.get(ids=chunk_ids)
            if not result or not result.get("ids"):
                return []

            lookup = {
                cid: {"id": cid, "document": doc, "metadata": meta}
                for cid, doc, meta in zip(
                    result.get("ids", []),
                    result.get("documents", []),
                    result.get("metadatas", []),
                )
            }
            return [lookup[cid] for cid in chunk_ids if cid in lookup]
        except Exception as exc:
            logger.error(f"[ChromaStore] get_many_by_ids failed for '{collection_name}' — {exc}")
            return []

    def query_children_with_parent_context(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search child records first, then return the matching parents."""
        raw = self.query_collection(
            collection_name=CONFIG.COL_CHILD,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
        )

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        self._logger.info(
            "[ChromaStore] Child search complete",
            query_texts=query_texts,
            requested=n_results,
            returned=len(ids),
            where=where,
        )

        best_children: dict[str, dict[str, Any]] = {}
        parent_order: list[str] = []

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            parent_id = (meta or {}).get("parent_id")
            if not parent_id:
                continue

            existing = best_children.get(parent_id)
            if existing is None or dist < existing["distance"]:
                best_children[parent_id] = {
                    "child_id": cid,
                    "child_text": doc,
                    "child_metadata": meta,
                    "distance": dist,
                }
            if parent_id not in parent_order:
                parent_order.append(parent_id)

        parents = self.get_many_by_ids(CONFIG.COL_PARENT, parent_order)
        parent_lookup = {rec["id"]: rec for rec in parents}

        self._logger.info(
            "[ChromaStore] Parent expansion complete",
            parent_ids=parent_order,
            parent_count=len(parents),
        )

        merged: list[dict[str, Any]] = []
        for parent_id in parent_order:
            parent = parent_lookup.get(parent_id)
            child = best_children.get(parent_id)
            if not parent or not child:
                continue
            merged.append(
                {
                    "id": parent["id"],
                    "text": parent["document"],
                    "metadata": parent["metadata"],
                    "parent_id": parent["id"],
                    "parent_text": parent["document"],
                    "parent_metadata": parent["metadata"],
                    "child_id": child["child_id"],
                    "child_text": child["child_text"],
                    "child_metadata": child["child_metadata"],
                    "distance": child["distance"],
                }
            )

        self._logger.debug(
            "[ChromaStore] Retrieval bundle",
            items=[
                {
                    "parent_id": item["parent_id"],
                    "child_id": item["child_id"],
                    "distance": item["distance"],
                }
                for item in merged
            ],
        )

        return merged

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