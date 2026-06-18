"""SQLite schema for query-answer cache memory."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

QUERY_CACHE_TABLE = "query_cache"

CACHE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS {QUERY_CACHE_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key TEXT NOT NULL UNIQUE,
        normalized_question TEXT NOT NULL,
        original_question TEXT NOT NULL,
        company TEXT,
        report_year TEXT,
        doc_type TEXT,
        extra_filters_json TEXT,
        top_k INTEGER NOT NULL,
        pipeline_version TEXT NOT NULL,
        status TEXT NOT NULL,
        answer_text TEXT NOT NULL,
        response_json TEXT NOT NULL,
        debug_json_path TEXT,
        hit_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_accessed_at TEXT,
        expires_at TEXT
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_query_cache_key ON {QUERY_CACHE_TABLE}(cache_key)",
    f"CREATE INDEX IF NOT EXISTS idx_query_cache_company_year ON {QUERY_CACHE_TABLE}(company, report_year)",
    f"CREATE INDEX IF NOT EXISTS idx_query_cache_status ON {QUERY_CACHE_TABLE}(status)",
)
