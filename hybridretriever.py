
# =============================================================================
# CELL 15 — Hybrid Retriever
# =============================================================================
"""
HybridRetriever: combines semantic search (ChromaDB) with BM25 keyword search
using Reciprocal Rank Fusion (RRF). Weights shift based on query intent type.
"""

from typing import Dict, List, Optional, Tuple
from logger import get_logger
from config import CONFIG
from chroma import CHROMA_STORE
from bm25 import BM25_INDEX


class HybridRetriever:
    """
    Retrieves chunks using a fusion of semantic and keyword search.

    Semantic weight vs BM25 weight shifts by query category:
      - factual_numerical / trend_temporal : 0.3 / 0.7
      - strategic / causal / risk          : 0.8 / 0.2
      - others                             : 0.5 / 0.5
    """

    # Weight presets per query category
    _WEIGHT_MAP: Dict[str, Tuple[float, float]] = {
        "factual_numerical": (0.3, 0.7),
        "trend_temporal": (0.3, 0.7),
        "strategic": (0.8, 0.2),
        "causal": (0.8, 0.2),
        "risk": (0.7, 0.3),
        "comparative": (0.5, 0.5),
        "general": (0.5, 0.5),
    }

    def __init__(self):
        """Initialise retriever with shared store and BM25 index."""
        self._logger = get_logger("retriever")
        self._chroma = CHROMA_STORE
        self._bm25 = BM25_INDEX

    def retrieve(
        self,
        query: str,
        collections: List[str],
        query_category: str = "general",
        scrip: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid semantic + BM25 search.

        Parameters
        ----------
        query : str
            Search query.
        collections : List[str]
            ChromaDB collections to search semantically.
        query_category : str
            Query type for weight selection.
        scrip : str, optional
            Metadata filter.
        fiscal_year : str, optional
            Metadata filter.
        top_k : int, optional
            Override CONFIG.FINAL_TOP_K.

        Returns
        -------
        List[Dict]
            Ranked dicts with: id, text, metadata, rrf_score.
        """
        final_k = top_k or CONFIG.FINAL_TOP_K
        sem_w, bm25_w = self._WEIGHT_MAP.get(query_category, (0.5, 0.5))

        sem_results = self._semantic_search(query, collections, scrip, fiscal_year)
        bm25_results = self._bm25_search(query, scrip, fiscal_year)
        merged = self._rrf_merge(sem_results, bm25_results, sem_w, bm25_w)
        return merged[:final_k]

    def get_parent_context(self, parent_id: str) -> Optional[Dict]:
        """
        Fetch a parent section chunk by its ID.

        Parameters
        ----------
        parent_id : str
            Parent chunk ID.

        Returns
        -------
        dict or None
        """
        return self._chroma.get_by_id(CONFIG.COL_PARENT, parent_id)

    # ── Private methods ───────────────────────────────────────────────────

    def _semantic_search(
        self,
        query: str,
        collections: List[str],
        scrip: Optional[str],
        fiscal_year: Optional[str],
    ) -> List[Dict]:
        """
        Query one or more ChromaDB collections and merge results.

        Parameters
        ----------
        query : str
        collections : List[str]
        scrip : str or None
        fiscal_year : str or None

        Returns
        -------
        List[Dict]
            Each dict: id, text, metadata, distance.
        """
        where: Optional[Dict] = None
        filters = {}
        if scrip:
            filters["scrip"] = scrip
        if fiscal_year:
            filters["fiscal_year"] = fiscal_year
        if filters:
            if len(filters) == 1:
                where = filters
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        results: List[Dict] = []
        for col_name in collections:
            try:
                raw = self._chroma.query_collection(
                    col_name, [query], CONFIG.SEMANTIC_TOP_K, where
                )
                ids = raw.get("ids", [[]])[0]
                docs = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas", [[]])[0]
                dists = raw.get("distances", [[]])[0]
                for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                    results.append({
                        "id": cid,
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                    })
            except Exception as exc:
                self._logger.warning(
                    "Semantic search failed for collection",
                    collection=col_name,
                    error=str(exc),
                )
        return results

    def _bm25_search(
        self,
        query: str,
        scrip: Optional[str],
        fiscal_year: Optional[str],
    ) -> List[Dict]:
        """
        BM25 keyword search with optional metadata filters.

        Parameters
        ----------
        query : str
        scrip : str or None
        fiscal_year : str or None

        Returns
        -------
        List[Dict]
        """
        return self._bm25.search(query, CONFIG.BM25_TOP_K, scrip, fiscal_year)

    def _rrf_merge(
        self,
        sem_results: List[Dict],
        bm25_results: List[Dict],
        sem_weight: float,
        bm25_weight: float,
    ) -> List[Dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Parameters
        ----------
        sem_results : List[Dict]
        bm25_results : List[Dict]
        sem_weight : float
        bm25_weight : float

        Returns
        -------
        List[Dict]
            Deduplicated, descending by rrf_score.
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, item in enumerate(sem_results):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sem_weight / (CONFIG.RRF_K + rank + 1)
            doc_map[cid] = item

        for rank, item in enumerate(bm25_results):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (CONFIG.RRF_K + rank + 1)
            if cid not in doc_map:
                doc_map[cid] = item

        merged = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = dict(doc_map[cid])
            entry["rrf_score"] = score
            merged.append(entry)
        return merged


RETRIEVER = HybridRetriever()

# ----------------------------------------------------------------------------
# Cell 15: Hybrid Retriever
# Purpose: Fuse semantic + BM25 retrieval using weighted RRF.
# Key Classes: HybridRetriever
# Key Functions:
#   HybridRetriever.retrieve(query, collections, category, scrip, fy, top_k) → List[Dict]
#   HybridRetriever.get_parent_context(parent_id) → dict | None
#   HybridRetriever._semantic_search(...) → List[Dict]
#   HybridRetriever._bm25_search(...) → List[Dict]
#   HybridRetriever._rrf_merge(...) → List[Dict]
# Key Constants/Config: CONFIG.RRF_K, CONFIG.SEMANTIC_TOP_K, CONFIG.BM25_TOP_K,
#   CONFIG.FINAL_TOP_K, _WEIGHT_MAP
# Imports exported: HybridRetriever, RETRIEVER
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 8 (CHROMA_STORE),
#   Cell 14 (BM25_INDEX)
# Critical notes: ChromaDB `where` filter requires a single-key dict or
#   {"$and": [...]} for multi-key. Tested both cases here.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

