# =============================================================================
# CELL 7 — Embedding Model
# =============================================================================
"""
LocalEmbedder: wraps all-MiniLM-L6-v2 via sentence-transformers.
Implements the ChromaDB EmbeddingFunction interface.
No API calls, no quota concerns. Singleton pattern.
"""

from typing import List, Optional
import numpy as np
from logger import get_logger
from config import Config

class LocalEmbedder:
    """
    Local embedding model wrapping all-MiniLM-L6-v2.

    Implements ChromaDB's EmbeddingFunction protocol so it can be passed
    directly to chromadb.Collection constructors.

    Singleton — use get_instance() rather than constructing directly.
    """

    _instance: Optional["LocalEmbedder"] = None
    _model = None

    def __init__(self):
        """Load the sentence-transformer model once."""
        self._logger = get_logger("embedder")
        self._CONFIG = Config.get_instance()
        self._load_model()

    def name(self) -> str:
        """Return the embedding function identifier for ChromaDB compatibility."""
        return self._CONFIG.EMBEDDING_MODEL  # ← THIS IS THE FIX

    def _load_model(self) -> None:
        """Load all-MiniLM-L6-v2 from sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._CONFIG.EMBEDDING_MODEL)
            self._logger.info(
                "Embedding model loaded.", model=self._CONFIG.EMBEDDING_MODEL, dim=self._CONFIG.EMBEDDING_DIM
            )
        except Exception as exc:
            self._logger.error("Failed to load embedding model", error=str(exc))
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts — called by ChromaDB on store and query.

        Parameters
        ----------
        input : List[str]
            Texts to embed.

        Returns
        -------
        List[List[float]]
            List of 384-dimensional float vectors.
        """
        if not input:
            return []
        try:
            vectors: np.ndarray = self._model.encode(
                input, show_progress_bar=False, convert_to_numpy=True
            )
            return vectors.tolist()
        except Exception as exc:
            self._logger.error("Embedding failed", error=str(exc), batch_size=len(input))
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string for retrieval.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        List[float]
            384-dimensional float vector.
        """
        return self([text])[0]

    @classmethod
    def get_instance(cls) -> "LocalEmbedder":
        """
        Return the singleton LocalEmbedder, loading the model if needed.

        Returns
        -------
        LocalEmbedder
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Load at import time so subsequent cells don't wait
EMBEDDER = LocalEmbedder.get_instance()


# ----------------------------------------------------------------------------
# Cell 7: Embedding Model
# Purpose: Provide a singleton sentence-transformer embedder for ChromaDB.
# Key Classes: LocalEmbedder
# Key Functions:
#   LocalEmbedder.__call__(input: List[str]) → List[List[float]]
#   LocalEmbedder.embed_query(text: str) → List[float]
#   LocalEmbedder.get_instance() → LocalEmbedder
# Key Constants/Config: CONFIG.EMBEDDING_MODEL, CONFIG.EMBEDDING_DIM
# Imports exported: LocalEmbedder, EMBEDDER
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: EMBEDDER is the shared singleton — pass it to ChromaDB
#   collection constructors as the embedding_function parameter.
#   __call__ signature matches chromadb.api.types.EmbeddingFunction.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
