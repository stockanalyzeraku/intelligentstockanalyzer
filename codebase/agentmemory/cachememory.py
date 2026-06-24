"""SQLite-backed query cache for financial RAG answers."""

from __future__ import annotations
import sys
import os

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from codebase.agentmemory.cachestructure import CACHE_SCHEMA_STATEMENTS, QUERY_CACHE_TABLE
from config import CONFIG

class CacheMemory:
    """Persist and retrieve RAG responses by deterministic query cache keys."""

    DEFAULT_PIPELINE_VERSION = "financial_rag_v1"

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        default_ttl_seconds: int | None = 60 * 60 * 24 * 30,
        pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    ) -> None:
        default_db_path = CONFIG.CACHE_MEMORY_DB_PATH
        self.db_path = Path(db_path or default_db_path)
        self.default_ttl_seconds = default_ttl_seconds
        self.pipeline_version = pipeline_version
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise_database()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with dictionary-like rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise_database(self) -> None:
        """Create cache tables and indexes if they do not exist."""
        with self.connect() as conn:
            for statement in CACHE_SCHEMA_STATEMENTS:
                conn.execute(statement)

    def build_cache_key(
        self,
        question: str,
        company: str | None = None,
        year: int | str | None = None,
        doc_type: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> tuple[str, dict[str, Any]]:
        """Return a stable hash and normalized payload for a planned query."""
        payload = {
            "question": self.normalize_question(question),
            "company": self._normalize_optional(company),
            "year": self._normalize_optional(year),
            "doc_type": self._normalize_optional(doc_type),
            "extra_filters": self._normalize_filters(extra_filters),
            "top_k": top_k,
            "pipeline_version": self.pipeline_version,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload

    def build_structured_cache_key(
        self,
        company: str | None = None,
        doc_type: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> tuple[str, dict[str, Any]]:
        """Phrasing-independent variant of build_cache_key for the multi-agent
        pipeline (codebase/agent/pipeline.py).

        ADDITIVE ONLY - build_cache_key above is completely unchanged, and
        every other method on this class is unaffected. This method exists
        because build_cache_key always hashes the raw question text as part
        of the cache identity, which means two different phrasings of a
        question that resolve to the SAME company/line_items/periods (e.g.
        "Sales for Kalyan Jewellers in 2023" vs "what was the sales figure
        for Kalyan Jewellers in 2023") would never share a cache entry.

        Callers should put everything that defines the query's *resolved*
        identity into extra_filters (e.g. line_items, periods,
        needs_qualitative_context, intent) - NOT raw question text. The
        literal question the user typed is never hashed by this method; it
        should still be passed as `original_question` to
        set_cached_response() for display/debugging, exactly as before.

        Returns the same (cache_key, normalized_payload) shape as
        build_cache_key, so it is a drop-in replacement at call sites:
        normalized_payload always includes a "question" key (a fixed
        placeholder string, since set_cached_response() and the
        normalized_question NOT NULL column both require one), but that
        placeholder is constant and contributes nothing to differentiate
        one cache entry from another - only company/doc_type/extra_filters/
        top_k/pipeline_version do.
        """
        payload = {
            "question": "structured_query",
            "company": self._normalize_optional(company),
            "year": None,
            "doc_type": self._normalize_optional(doc_type),
            "extra_filters": self._normalize_filters(extra_filters),
            "top_k": top_k,
            "pipeline_version": self.pipeline_version,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload

    def get_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached response payload and update hit metadata when present."""
        now = self._now()
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {QUERY_CACHE_TABLE} WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None

            row_dict = dict(row)
            expires_at = row_dict.get("expires_at")
            if expires_at and expires_at <= now:
                conn.execute(f"DELETE FROM {QUERY_CACHE_TABLE} WHERE cache_key = ?", (cache_key,))
                return None

            conn.execute(
                f"""
                UPDATE {QUERY_CACHE_TABLE}
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE cache_key = ?
                """,
                (now, cache_key),
            )

        row_dict["hit_count"] = int(row_dict.get("hit_count") or 0) + 1
        row_dict["response"] = json.loads(row_dict["response_json"])
        row_dict["extra_filters"] = self._loads(row_dict.get("extra_filters_json"))
        return row_dict

    def set_cached_response(
        self,
        cache_key: str,
        normalized_payload: dict[str, Any],
        original_question: str,
        response: Any,
        debug_json_path: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert or update a cached RAG response."""
        response_dict = asdict(response) if is_dataclass(response) else dict(response)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = self._expiry(ttl)
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {QUERY_CACHE_TABLE} (
                    cache_key, normalized_question, original_question, company,
                    report_year, doc_type, extra_filters_json, top_k,
                    pipeline_version, status, answer_text, response_json,
                    debug_json_path, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    status = excluded.status,
                    answer_text = excluded.answer_text,
                    response_json = excluded.response_json,
                    debug_json_path = excluded.debug_json_path,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    normalized_payload["question"],
                    original_question,
                    normalized_payload.get("company"),
                    normalized_payload.get("year"),
                    normalized_payload.get("doc_type"),
                    json.dumps(normalized_payload.get("extra_filters", {}), sort_keys=True, default=str),
                    int(normalized_payload.get("top_k", 8)),
                    normalized_payload.get("pipeline_version", self.pipeline_version),
                    response_dict.get("status", "unknown"),
                    response_dict.get("answer", ""),
                    json.dumps(response_dict, ensure_ascii=False, default=str),
                    debug_json_path or response_dict.get("debug_json_path"),
                    expires_at,
                ),
            )

    def delete_expired(self) -> int:
        """Delete expired cache entries and return the number of deleted rows."""
        with self.connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {QUERY_CACHE_TABLE} WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (self._now(),),
            )
            return cursor.rowcount

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize whitespace and case for exact cache matching."""
        return " ".join(question.strip().lower().split())

    @staticmethod
    def _normalize_optional(value: Any) -> str | None:
        return str(value).strip().upper() if value not in (None, "") else None

    @staticmethod
    def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
        if not filters:
            return {}
        return {str(key): filters[key] for key in sorted(filters)}

    @staticmethod
    def _loads(value: str | None) -> Any:
        return json.loads(value) if value else None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _expiry(cls, ttl_seconds: int | None) -> str | None:
        if ttl_seconds is None:
            return None
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
