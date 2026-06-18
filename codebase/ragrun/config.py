"""Configuration helpers for the background RAG worker."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass
from pathlib import Path

from config import CONFIG as BASE_CONFIG


@dataclass(frozen=True)
class RAGRunConfig:
    """Small adapter around the project's base Config.py settings."""

    chroma_path: str = BASE_CONFIG.CHROMA_PATH
    database_path: str = BASE_CONFIG.DB_PATH
    debug_output_dir: str = str(Path(BASE_CONFIG.LOGS_PATH) / "ragrun_debug")
    collection_name: str = BASE_CONFIG.COL_CHILD
    top_k: int = BASE_CONFIG.FINAL_TOP_K
    cache_ttl_seconds: int = 60 * 60 * 24 * 30
    mistral_model: str = "mistral-large-latest"
    gemini_model: str = BASE_CONFIG.GEMINI_MODEL or "gemini-1.5-pro"
    mistral_api_key: str | None = BASE_CONFIG.MISTRAL_API_KEY
    gemini_api_key: str | None = BASE_CONFIG.GEMINI_API_KEY
    temperature: float = 0.0
    max_output_tokens: int = BASE_CONFIG.LLM_MAX_OUTPUT_TOKENS
    timeout_seconds: int = BASE_CONFIG.LLM_TIMEOUT_SECONDS


RAGRUN_CONFIG = RAGRunConfig()
