# =============================================================================
# CELL 14 — BM25 Keyword Index
# =============================================================================
"""
BM25Index: keyword search over all stored chunks.
Thread-safe. Caps at BM25_MAX_DOCS to bound memory.
Must call build() after adding new chunks.
"""

import re
import string
import threading
from typing import Dict, List, Optional
from config import CONFIG
from chroma import CHROMA_STORE
from logger import get_logger


class BM25Index:
    """
    Keyword index using BM25Okapi over all ChromaDB child chunks.

    Thread-safe via RLock. Bound to BM25_MAX_DOCS documents.
    Must call build() after any new chunks are added to ChromaDB.
    """

    def __init__(self):
        """Initialise an empty BM25 index."""
        self._logger = get_logger("bm25_index")
        self._lock = threading.RLock()
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_metadatas: List[Dict] = []
        self._doc_texts: List[str] = []

    def build(self) -> None:
        """
        (Re)build the BM25 index from all current chunks in ChromaDB.

        Loads from child_chunks, financial_facts, and mgmt_statements.
        Caps at CONFIG.BM25_MAX_DOCS.
        """
        from rank_bm25 import BM25Okapi

        _logger = self._logger
        _logger.info("Building BM25 index...")
        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metas: List[Dict] = []

        collections_to_index = [CONFIG.COL_CHILD, CONFIG.COL_FACTS, CONFIG.COL_MGMT]
        for col_name in collections_to_index:
            try:
                col = CHROMA_STORE._collections.get(col_name)
                if col is None:
                    continue
                count = col.count()
                if count == 0:
                    continue
                result = col.get(
                    limit=min(count, CONFIG.BM25_MAX_DOCS - len(all_ids)),
                    include=["documents", "metadatas"],
                )
                all_ids.extend(result.get("ids", []))
                all_texts.extend(result.get("documents", []))
                all_metas.extend(result.get("metadatas", []))
                if len(all_ids) >= CONFIG.BM25_MAX_DOCS:
                    break
            except Exception as exc:
                _logger.error("Failed to load from collection", col=col_name, error=str(exc))

        if not all_texts:
            _logger.warning("No documents found for BM25 build.")
            return

        tokenised = [self._tokenise(t) for t in all_texts]
        with self._lock:
            self._bm25 = BM25Okapi(tokenised)
            self._doc_ids = all_ids
            self._doc_metadatas = all_metas
            self._doc_texts = all_texts

        _logger.info("BM25 index built.", doc_count=len(all_ids))

    def search(
        self,
        query: str,
        top_k: int = 10,
        scrip: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> List[Dict]:
        """
        Keyword search with optional metadata filters.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Maximum results to return.
        scrip : str, optional
            Filter to a specific company scrip.
        fiscal_year : str, optional
            Filter to a specific fiscal year.

        Returns
        -------
        List[Dict]
            Dicts with keys: id, text, metadata, score.
        """
        with self._lock:
            if self._bm25 is None:
                self._logger.warning("BM25 index not built yet.")
                return []

            tokens = self._tokenise(query)
            scores = self._bm25.get_scores(tokens)

        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self._doc_metadatas[idx]
            if scrip and meta.get("scrip", "").upper() != scrip.upper():
                continue
            if fiscal_year and meta.get("fiscal_year", "") != fiscal_year:
                continue
            results.append({
                "id": self._doc_ids[idx],
                "text": self._doc_texts[idx],
                "metadata": meta,
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Tokenise text: lowercase, remove punctuation, min length 2.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[str]
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [w for w in text.split() if len(w) >= 2]


BM25_INDEX = BM25Index()
# Note: BM25_INDEX.build() is called by pipeline after storing chunks.

# ----------------------------------------------------------------------------
# Cell 14: BM25 Keyword Index
# Purpose: Keyword retrieval with scrip/FY metadata filters via BM25Okapi.
# Key Classes: BM25Index
# Key Functions:
#   BM25Index.build() → None
#   BM25Index.search(query, top_k, scrip, fiscal_year) → List[Dict]
#   BM25Index._tokenise(text) → List[str]
# Key Constants/Config: CONFIG.BM25_MAX_DOCS, CONFIG.COL_CHILD/FACTS/MGMT
# Imports exported: BM25Index, BM25_INDEX
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 8 (CHROMA_STORE)
# Critical notes: build() must be called after every pipeline run.
#   BM25_INDEX is the global singleton — do not create additional instances.
#   Index is held in memory; large corpora may require RAM monitoring.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

