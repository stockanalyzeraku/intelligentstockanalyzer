# =============================================================================
# CELL 8 — ChromaDB Store
# =============================================================================
"""
ChromaStore: single access point for all four ChromaDB collections.
Wraps every external call with the CB_CHROMADB circuit breaker.
Singleton pattern.
"""

from typing import Dict, List, Optional, Any
import chromadb
from logger import get_logger
from embedder import LocalEmbedder
from config import Config

class ChromaStore:
    """
    Manages the four ChromaDB collections used by the agent.

    Collections
    -----------
    child_chunks    : 400-token paragraphs (primary retrieval)
    parent_sections : 2500-token full sections (context expansion)
    financial_facts : atomic numeric facts
    mgmt_statements : chairman letter / MD&A / strategy content

    Singleton — use get_instance().
    """

    _instance: Optional["ChromaStore"] = None

    def __init__(self):
        """Initialise ChromaDB client and ensure all collections exist."""
        self._CONFIG = Config.get_instance()
        self._logger = get_logger("chroma_store")
        self._client = chromadb.PersistentClient(path=self._CONFIG.CHROMA_PATH)
        self._embedder = LocalEmbedder.get_instance()
        self._collections: Dict[str, chromadb.Collection] = {}
        self._init_collections()


    def _init_collections(self) -> None:
        """Create or retrieve all four collections."""
        names = [self._CONFIG.COL_CHILD, self._CONFIG.COL_PARENT, self._CONFIG.COL_FACTS, self._CONFIG.COL_MGMT]
        for name in names:
            col = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedder,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[name] = col
            self._logger.info("Collection ready.", collection=name, count=col.count())

    def query_collection(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query a collection with circuit-breaker protection.

        Parameters
        ----------
        collection_name : str
            One of the four collection name constants from Config.
        query_texts : List[str]
            Query strings to embed and search.
        n_results : int
            Maximum results to return per query.
        where : dict, optional
            ChromaDB metadata filter.

        Returns
        -------
        dict
            ChromaDB query result dict with keys: ids, documents, metadatas, distances.
        """
        col = self._collections.get(collection_name)
        if col is None:
            self._logger.error("Unknown collection", name=collection_name)
            raise ValueError(f"Unknown collection: {collection_name}")

        def _do_query():
            kwargs: Dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": min(n_results, max(col.count(), 1)),
            }
            if where:
                kwargs["where"] = where
            return col.query(**kwargs)

        try:
            return CB_CHROMADB.call(_do_query)
        except CircuitBreakerOpenError as exc:
            self._logger.error("ChromaDB circuit open", error=str(exc))
            raise
        except Exception as exc:
            self._logger.error("ChromaDB query failed", collection=collection_name, error=str(exc))
            raise

    def upsert_batch(self, collection_name: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        """
        Idempotent batch upsert into a collection.

        Parameters
        ----------
        collection_name : str
            Target collection name.
        ids : List[str]
            Chunk IDs (format: {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}).
        documents : List[str]
            Text content of each chunk.
        metadatas : List[Dict]
            Metadata dicts (one per chunk).
        """
        if not ids:
            return
        col = self._collections.get(collection_name)
        if col is None:
            raise ValueError(f"Unknown collection: {collection_name}")

        # Process in batches of CONFIG.DB_BATCH_SIZE
        for start in range(0, len(ids), self._CONFIG.DB_BATCH_SIZE):
            batch_ids = ids[start: start + self._CONFIG.DB_BATCH_SIZE]
            batch_docs = documents[start: start + self._CONFIG.DB_BATCH_SIZE]
            batch_meta = metadatas[start: start + self._CONFIG.DB_BATCH_SIZE]
            try:
                CB_CHROMADB.call(col.upsert, ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            except Exception as exc:
                self._logger.error(
                    "Upsert batch failed",
                    collection=collection_name,
                    batch_start=start,
                    error=str(exc),
                )
                raise
        self._logger.info("Upsert complete.", collection=collection_name, total=len(ids))

    def get_by_id(self, collection_name: str, chunk_id: str) -> Optional[Dict]:
        """
        Fetch a single chunk by its ID.

        Parameters
        ----------
        collection_name : str
            Collection to search.
        chunk_id : str
            Exact chunk ID.

        Returns
        -------
        dict or None
            {'id', 'document', 'metadata'} or None if not found.
        """
        col = self._collections.get(collection_name)
        if col is None:
            return None
        try:
            result = CB_CHROMADB.call(col.get, ids=[chunk_id])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as exc:
            self._logger.error("get_by_id failed", id=chunk_id, error=str(exc))
        return None

    def status(self) -> str:
        """
        Return a human-readable summary of chunk counts per collection.

        Returns
        -------
        str
        """
        lines = ["=== ChromaDB Status ==="]
        for name, col in self._collections.items():
            lines.append(f"  {name}: {col.count()} chunks")
        return "\n".join(lines)

    @classmethod
    def get_instance(cls) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


CHROMA_STORE = ChromaStore.get_instance()
print(CHROMA_STORE.status())

# ----------------------------------------------------------------------------
# Cell 8: ChromaDB Store
# Purpose: Manage all ChromaDB collections with circuit-breaker protection.
# Key Classes: ChromaStore
# Key Functions:
#   ChromaStore.query_collection(name, query_texts, n_results, where) → dict
#   ChromaStore.upsert_batch(name, ids, documents, metadatas) → None
#   ChromaStore.get_by_id(name, chunk_id) → dict | None
#   ChromaStore.status() → str
#   ChromaStore.get_instance() → ChromaStore
# Key Constants/Config: CONFIG.COL_CHILD/PARENT/FACTS/MGMT, CONFIG.CHROMA_PATH,
#   CONFIG.DB_BATCH_SIZE, CB_CHROMADB
# Imports exported: ChromaStore, CHROMA_STORE
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 5 (CB_CHROMADB,
#   CircuitBreakerOpenError), Cell 7 (EMBEDDER)
# Critical notes: All four collections are created at init time.
#   query_collection caps n_results to col.count() to avoid ChromaDB error
#   when the collection has fewer items than n_results.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------


