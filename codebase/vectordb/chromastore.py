from __future__ import annotations
import os
import sys

import json
from typing import Any, Optional
import chromadb

from config import CONFIG
from logger import get_logger
from inputvalidator import InputValidator
from healthcheck import assert_system_health
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
        self.chroma_path = chroma_path or CONFIG.CHROMA_DB_PATH
        self._client: chromadb.PersistentClient | None = None
        self._logger = logger
        logger.info(f"[ChromaStore] Initialised — path='{self.chroma_path}'")

    
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
    
    #Storage in Chroma DB (Functions)

    def _upsert_records(self, collection_name: str, records: list[dict]) -> None:
        """Upsert a list of records into one collection."""
        if not records:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for record in records:
            ids.append(record["id"])
            documents.append(record["text"])
            metadatas.append(self._sanitize_metadata(record.get("metadata", {})))

        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, and metadatas must have matching lengths.")
        if not ids:
            logger.warning("[ChromaStore] Empty ids, SKIPPING.")
            return

        self.upsert_batch(collection_name, ids, documents, metadatas)


    def upsert_batch(self, collection_name : str, ids:list[str], documents:list[str], metadatas:list[dict], batch_size:int = 100) -> None:
        """
        Idempotent batch upsert — safe to re-run without duplicate errors.
        """
        #We are getting collection again anad again
        collection = self._get_collection(collection_name)

        for start in range(0, len(ids), batch_size):
            batch_ids  = ids      [start : start + batch_size]
            batch_docs = documents[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            try:
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
                logger.process_event("chroma_upsert_batch_completed", "vectordb", collection=collection_name, start=start, count=len(batch_ids))
            except Exception as exc:
                logger.exception("Chroma upsert batch failed", exc, event="chroma_upsert_batch_failed", collection=collection_name, start=start)
                raise

        logger.info(f"[ChromaStore] Upsert complete — {len(ids)} chunks into '{collection_name}'")


    def store_in_chromadb(self, embedding_json_path : str, collection_name:str, chroma_path:str | None = None) -> chromadb.Collection:
        """
        Loads records from a JSON file and upserts them into ChromaDB.
        """
        #Checks Chroma DB Health
        assert_system_health(include_chroma=False)

        collection_name = InputValidator.validate_collection_name(collection_name)
        embedding_json_path = InputValidator.validate_json_path(embedding_json_path, must_exist=True)
        if chroma_path:
            self.chroma_path = chroma_path
            self._client     = None

        logger.process_event("chroma_store_started", "vectordb", collection=collection_name, path=self.chroma_path)

        with open(embedding_json_path, "r", encoding="utf-8") as fh:
            payload: Any = json.load(fh)
        payload = InputValidator.validate_embedding_payload(payload)

        if isinstance(payload, dict) and "parents" in payload and "children" in payload:
            parents = payload.get("parents", [])
            children = payload.get("children", [])
            #Validation can be added on children and parent
            logger.process_event("embedding_bundle_loaded", "vectordb", parents=len(parents), children=len(children))
            self._upsert_records(CONFIG.COL_PARENT, parents)
            self._upsert_records(CONFIG.COL_CHILD, children)
            logger.process_event("chroma_store_completed", "vectordb", parent_collection=CONFIG.COL_PARENT, child_collection=CONFIG.COL_CHILD, parents=len(parents), children=len(children))
            return self._get_collection(CONFIG.COL_CHILD)

        records: list[dict] = payload
        logger.process_event("embedding_records_loaded", "vectordb", records=len(records), collection=collection_name)
        self._upsert_records(collection_name, records)
        logger.process_event("chroma_store_completed", "vectordb", records=len(records), collection=collection_name)
        return self._get_collection(collection_name)

    # ------------------------------------------------------------------------------------------------------

    # def query_collection(
    #     self,
    #     collection_name : str,
    #     query_texts     : list[str],
    #     n_results       : int = 10,
    #     where           : dict | None = None,
    # ) -> dict[str, Any]:
    #     """
    #     Query a collection by text — ChromaDB embeds the query automatically.

    #     Parameters
    #     ----------
    #     collection_name : Collection to search.
    #     query_texts     : One or more query strings.
    #     n_results       : Max results to return.
    #     where           : Optional metadata filter e.g. {"company": "KALYANKJIL"}.

    #     Returns
    #     -------
    #     dict with keys: ids, documents, metadatas, distances
    #     """
    #     collection = self._get_collection(collection_name)
    #     kwargs: dict[str, Any] = {
    #         "query_texts" : query_texts,
    #         "n_results"   : min(n_results, max(collection.count(), 1)),
    #     }
    #     if where:
    #         kwargs["where"] = where

    #     try:
    #         return collection.query(**kwargs)
    #     except Exception as exc:
    #         logger.error(f"[ChromaStore] Query failed on '{collection_name}' — {exc}")
    #         raise

    # -------------------Query Collection--------------------
    
    def query_collection(self, collection_name: str, query_texts: list[str], n_results: int = 10, where: dict | None = None,) -> dict:
        collection_name = InputValidator.validate_collection_name(collection_name)
        query_texts = InputValidator.validate_query_texts(query_texts)
        where = InputValidator.validate_chroma_where(where)
        collection = self._get_collection(collection_name)
        n_results = InputValidator.validate_top_k(n_results, default=10, max_value=max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K))
        
        query_embeddings = [EMBEDDER.embed_query(q) for q in query_texts]

        logger.process_event("chroma_query_started", "vectordb", collection=collection_name, n_results=n_results, query_count=len(query_texts))
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        returned = len(result.get("ids", [[]])[0]) if result.get("ids") else 0
        logger.process_event("chroma_query_completed", "vectordb", collection=collection_name, returned=returned)
        return result

    def get_many_by_ids(self, collection_name: str, chunk_ids: list[str]) -> list[dict]:
        """Fetch multiple chunks by exact id and preserve the requested order."""

        collection_name = InputValidator.validate_collection_name(collection_name)
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

    def query_children_with_parent_context(self, query_texts: list[str], n_results: int = 1, where: dict | None = None,) -> list[dict[str, Any]]:
        """Search child records first, then return the matching parents."""
        
        raw = self.query_collection(collection_name=CONFIG.COL_CHILD, query_texts=query_texts, n_results=n_results, where=where,)
        
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        self._logger.info("[ChromaStore] Child search complete", query_texts=query_texts, requested=n_results, returned=len(ids), where=where)

        best_children: dict[str, dict[str, Any]] = {}
        parent_order: list[str] = []

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            parent_id = (meta or {}).get("parent_id")
            if not parent_id:
                continue
            #I feel this part has no logic to be here
            existing = best_children.get(parent_id)
            if existing is None or dist < existing["distance"]:
                best_children[parent_id] = {
                    "child_id": cid,
                    "child_text": doc,
                    "child_metadata": meta,
                    "distance": dist,
                }
            # Till this part
            if parent_id not in parent_order:
                parent_order.append(parent_id)

        parents = self.get_many_by_ids(CONFIG.COL_PARENT, parent_order)

        parent_lookup = {rec["id"]: rec for rec in parents}

        self._logger.info("[ChromaStore] Parent expansion complete", parent_ids=parent_order, parent_count=len(parents))

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

        self._logger.debug("[ChromaStore] Retrieval bundle",items=[{"parent_id": item["parent_id"],"child_id": item["child_id"],"distance": item["distance"]} for item in merged])

        return merged
    
    # --------------------For showing Status-----------------

    def status(self, collection_name: str) -> str:
        """
        Returns a summary of chunk count in a collection.
        """
        try:
            collection = self._get_collection(collection_name)
            count      = collection.count()
            summary    = f"{collection_name}: {count} chunks"
            logger.info(f"[ChromaStore] {summary}")
            return summary
        except Exception as exc:
            logger.error(f"[ChromaStore] status failed — {exc}")
            return f"Error fetching status: {exc}"
    
    # -------------------used for debugging------------------

    def get_by_id(self, collection_name : str, chunk_id:str) -> dict | None:
        """
        Fetch a single chunk by its exact ID.
        Used for debugging
        """
        collection_name = InputValidator.validate_collection_name(collection_name)
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
        
    @classmethod
    def get_instance(cls, chroma_path: str | None = None) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(chroma_path)
        return cls._instance

    
CHROMASTORE = ChromaStore()

if __name__ == "__main__":
    CHROMASTORE.get_instance()