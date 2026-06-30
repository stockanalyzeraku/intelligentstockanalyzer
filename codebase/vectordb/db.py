"""ChromaDB connection and collection lifecycle.

This is the only file in the module allowed to know how a Chroma
PersistentClient is constructed. It has no knowledge of parent/child
chunking, JSON payloads, or health checks — those are higher-level
concerns layered on top in store.py / retriever.py / chromastore.py.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import chromadb

from codebase.vectordb.skelton import HNSW_SPACE


class ChromaConnection:
    """Owns the single PersistentClient for a given on-disk Chroma path.

    True singleton: `ChromaConnection(path)` called twice always returns
    the same object, so nothing in the app can end up holding two live
    Chroma clients by accident.
    """

    _instance: Optional["ChromaConnection"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, chroma_path: str, embedding_function: Any = None):
        if getattr(self, "_initialized", False):
            return
        self._chroma_path = chroma_path
        self._embedding_function = embedding_function
        self._client: chromadb.ClientAPI | None = None
        self._initialized = True

    @property
    def chroma_path(self) -> str:
        return self._chroma_path

    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._chroma_path)
        return self._client

    def reset_path(self, chroma_path: str) -> None:
        """Point this connection at a different on-disk path."""
        self._chroma_path = chroma_path
        self._client = None

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        return self.client().get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": HNSW_SPACE},  # set once, can't change later
        )