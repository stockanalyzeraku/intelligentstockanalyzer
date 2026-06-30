"""System health checks for startup and pre-job guardrails."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from config import CONFIG
from errorhandler import CB_CHROMADB, CB_EMBEDDER, CB_FILESYSTEM, CB_GEMINI, CB_MISTRAL, CB_SQLITE
from logger import get_logger

logger = get_logger("healthcheck")


def _check_path(name: str, path: str, *, must_be_dir: bool = True, writable: bool = True) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {"status": "ok", "path": str(target)}
    try:
        if must_be_dir:
            target.mkdir(parents=True, exist_ok=True)
            if not target.is_dir():
                raise NotADirectoryError(str(target))
            probe_dir = target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            probe_dir = target.parent
        if writable:
            with tempfile.NamedTemporaryFile(prefix=f".{name}_health_", dir=probe_dir, delete=True) as fh:
                fh.write(b"ok")
                fh.flush()
        return result
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc), "error_type": exc.__class__.__name__})
        return result


def _check_disk(path: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_pct = round((usage.free / usage.total) * 100, 2) if usage.total else 0.0
    status = "ok" if free_pct >= 10 else "degraded" if free_pct >= 5 else "failed"
    return {
        "status": status,
        "path": path,
        "total_mb": round(usage.total / (1024 * 1024), 2),
        "free_mb": round(usage.free / (1024 * 1024), 2),
        "free_pct": free_pct,
    }


def _check_sqlite() -> dict[str, Any]:
    db_path = Path(CONFIG.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(str(db_path), timeout=3) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS healthcheck (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO healthcheck(value) VALUES (?)", ("ok",))
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            row = conn.execute("SELECT value FROM healthcheck WHERE id = ?", (row_id,)).fetchone()
            conn.execute("DELETE FROM healthcheck WHERE id = ?", (row_id,))
            conn.commit()
        return {"status": "ok" if row and row[0] == "ok" else "failed", "path": str(db_path)}
    except Exception as exc:
        return {"status": "failed", "path": str(db_path), "error": str(exc), "error_type": exc.__class__.__name__}


def _check_chroma(collection_name: str | None = None) -> dict[str, Any]:
    try:
        from codebase.vectordb.chromastore import ChromaStore

        store = ChromaStore.get_instance(CONFIG.CHROMA_DB_PATH)
        target_collection = collection_name or CONFIG.COL_CHILD
        count = store.collection_count(target_collection)
        return {"status": "ok", "path": CONFIG.CHROMA_DB_PATH, "collection": target_collection, "count": count}
    except Exception as exc:
        return {"status": "failed", "path": CONFIG.CHROMA_DB_PATH, "error": str(exc), "error_type": exc.__class__.__name__}


def _check_embedder() -> dict[str, Any]:
    try:
        from codebase.vectordb.embedder import EMBEDDER

        vector = EMBEDDER(["health check"])[0]
        dim = len(vector)
        return {"status": "ok" if dim == CONFIG.EMBEDDING_DIM else "degraded", "dimension": dim, "expected_dimension": CONFIG.EMBEDDING_DIM}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "error_type": exc.__class__.__name__}


def _check_llm_config() -> dict[str, Any]:
    providers = {
        "mistral": bool(CONFIG.MISTRAL_API_KEY),
        "gemini": bool(CONFIG.GEMINI_API_KEY),
    }
    status = "ok" if any(providers.values()) else "degraded"
    return {"status": status, "providers_configured": providers}


def system_health(*, include_chroma: bool = False, include_embedder: bool = False, include_llm: bool = False, collection_name: str | None = None) -> dict[str, Any]:
    """Return a structured health report for startup/pre-job checks.

    Expensive checks are opt-in: Chroma and embedder imports can load local model
    state, and LLM checks are intentionally limited to key/config readiness.
    """
    start = time.perf_counter()
    checks: dict[str, Any] = {
        "base_path": _check_path("base", CONFIG.BASE_PATH),
        "uploads_path": _check_path("uploads", CONFIG.UPLOADS_PATH),
        "logs_path": _check_path("logs", CONFIG.LOGS_PATH),
        "database_path": _check_path("database", CONFIG.DB_PATH, must_be_dir=False),
        "disk": _check_disk(CONFIG.BASE_PATH),
        "sqlite": _check_sqlite(),
        "circuit_breakers": {
            "gemini": CB_GEMINI.status(),
            "mistral": CB_MISTRAL.status(),
            "chromadb": CB_CHROMADB.status(),
            "embedder": CB_EMBEDDER.status(),
            "filesystem": CB_FILESYSTEM.status(),
            "sqlite": CB_SQLITE.status(),
        },
    }
    if include_chroma:
        checks["chroma"] = _check_chroma(collection_name)
    if include_embedder:
        checks["embedder"] = _check_embedder()
    if include_llm:
        checks["llm"] = _check_llm_config()

    statuses = [item.get("status", "ok") for item in checks.values() if isinstance(item, dict)]
    overall = "failed" if "failed" in statuses else "degraded" if "degraded" in statuses else "ok"
    report = {"status": overall, "duration_ms": round((time.perf_counter() - start) * 1000, 2), "checks": checks}
    logger.info("System health check completed", event="system_health_completed", status=overall, duration_ms=report["duration_ms"])
    return report


def assert_system_health(**kwargs: Any) -> dict[str, Any]:
    """Run system_health and raise RuntimeError on failed status."""
    report = system_health(**kwargs)
    if report["status"] == "failed":
        raise RuntimeError(f"System health failed: {report}")
    return report