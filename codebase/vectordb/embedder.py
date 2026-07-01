"""LocalEmbedder: wraps all-MiniLM-L6-v2 via sentence-transformers.

Implements ChromaDB's EmbeddingFunction protocol (a callable taking a
list of strings and returning a list of vectors) so it can be passed
directly to chromadb.Collection constructors, and called the same way
by anything else in the app that needs a vector for a single piece of
text — there's exactly one way to get an embedding out of this class.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from config import CONFIG
from logger import get_logger


class LocalEmbedder:
    """Local embedding model wrapping all-MiniLM-L6-v2.

    Singleton — use get_instance() rather than constructing directly.
    """

    _instance: Optional["LocalEmbedder"] = None
    _model = None

    def __init__(self):
        self._logger = get_logger("VECTORDB", "embedder")
        self._CONFIG = CONFIG
        self._load_model()

    def name(self) -> str:
        """Embedding function identifier, for ChromaDB compatibility."""
        return self._CONFIG.EMBEDDING_MODEL

    def _load_model(self) -> None:
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
        """Embed a batch of texts. Used for both documents and queries —
        pass a single-item list to embed one piece of text."""
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

    @classmethod
    def get_instance(cls) -> "LocalEmbedder":
        """Return the singleton LocalEmbedder, loading the model if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Loaded once at import time so every caller shares the same model in memory.
EMBEDDER = LocalEmbedder.get_instance()