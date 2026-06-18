"""Cache adapter for the background RAG worker."""

from __future__ import annotations

from typing import Any

from codebase.agentmemory.cachememory import CacheMemory
from codebase.ragrun.config import RAGRUN_CONFIG
from codebase.ragrun.schemas import RAGWorkerResponse


class RAGCacheManager:
    """Read and write normalized-query answers using the existing cache module."""

    def __init__(self, cache_memory: CacheMemory | None = None) -> None:
        self.cache = cache_memory or CacheMemory(
            db_path=RAGRUN_CONFIG.database_path,
            default_ttl_seconds=RAGRUN_CONFIG.cache_ttl_seconds,
            pipeline_version="ragrun_v1",
        )

    def build_key(self, query: str, top_k: int) -> tuple[str, dict[str, Any]]:
        return self.cache.build_cache_key(question=query, top_k=top_k)

    def get(self, cache_key: str) -> dict[str, Any] | None:
        return self.cache.get_cached_response(cache_key)

    def set(self, cache_key: str, payload: dict[str, Any], query: str, response: RAGWorkerResponse) -> None:
        self.cache.set_cached_response(
            cache_key=cache_key,
            normalized_payload=payload,
            original_question=query,
            response=response.to_dict(),
            debug_json_path=response.debug_json_path,
        )
