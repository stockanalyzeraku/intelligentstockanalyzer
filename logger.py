# =============================================================================
# CELL 4 — Logger
# =============================================================================
"""
Structured JSON logger with run context, redaction, duration helpers, and
exception logging. Every module creates its own StructuredLogger instance.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, Mapping, Optional
from uuid import uuid4

from config import CONFIG


_SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
_MAX_FIELD_CHARS = 2_000


class StructuredLogger:
    """
    Structured JSON logger for the Investment Brain Agent.

    Writes every log entry as a single-line JSON object to a daily component log
    file. Optional bound context (run_id, job_id, company, etc.) is included in
    every entry from child loggers created with :meth:`bind`.
    """

    def __init__(self, component: str, config=None, context: Optional[Mapping[str, Any]] = None):
        """Initialise logger for the given component."""
        self._component = component
        self._config = config or CONFIG
        self._context: Dict[str, Any] = dict(context or {})
        self._ist = timezone(timedelta(hours=5, minutes=30))
        self._log_file = self._resolve_log_path()
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)

    # ── Context helpers ──────────────────────────────────────────────────

    def bind(self, **context: Any) -> "StructuredLogger":
        """Return a child logger that adds *context* to every log entry."""
        merged = {**self._context, **context}
        return StructuredLogger(self._component, self._config, merged)

    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        """Create a compact unique run id for correlating logs."""
        return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    # ── Public API ────────────────────────────────────────────────────────

    def debug(self, message: str, **kwargs: Any) -> None:
        self._write("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._write("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._write("WARNING", message, echo=True, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._write("ERROR", message, echo=True, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._write("CRITICAL", message, echo=True, **kwargs)

    def exception(self, message: str, exc: BaseException, **kwargs: Any) -> None:
        """Log an exception with type, message, and traceback."""
        self.error(
            message,
            event=kwargs.pop("event", "exception"),
            exception_type=exc.__class__.__name__,
            exception_message=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            **kwargs,
        )

    def process_event(self, event: str, stage: str, status: str = "ok", **fields: Any) -> None:
        """Log a normalized pipeline/process lifecycle event."""
        level = "ERROR" if status == "failed" else "WARNING" if status in {"warning", "degraded", "skipped"} else "INFO"
        self._write(
            level,
            fields.pop("message", event.replace("_", " ").title()),
            echo=level in {"ERROR", "WARNING"},
            event=event,
            stage=stage,
            status=status,
            **fields,
        )

    @contextmanager
    def timed(self, event: str, **fields: Any) -> Iterator[None]:
        """Log start/completion/failure events with duration_ms."""
        start = time.perf_counter()
        self.info(f"{event} started", event=f"{event}_started", **fields)
        try:
            yield
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.exception(
                f"{event} failed",
                exc,
                event=f"{event}_failed",
                duration_ms=duration_ms,
                **fields,
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.info(
                f"{event} completed",
                event=f"{event}_completed",
                duration_ms=duration_ms,
                **fields,
            )

    # ── Private helpers ───────────────────────────────────────────────────

    def _resolve_log_path(self) -> str:
        """Compute today's log file path."""
        date_str = datetime.now(self._ist).strftime("%Y-%m-%d")
        safe_component = str(self._component).replace(os.sep, ".")
        filename = f"{date_str}_{safe_component}.log"
        return os.path.join(self._config.LOGS_PATH, filename)

    def _write(self, level: str, message: str, echo: bool = False, **kwargs: Any) -> None:
        entry: Dict[str, Any] = {
            "ts": datetime.now(self._ist).isoformat(),
            "level": level,
            "component": self._component,
            "message": message,
        }
        entry.update(self._context)
        entry.update(kwargs)
        line = json.dumps(self._sanitize(entry), default=str, ensure_ascii=False)
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            print(f"[Logger write failure] {exc} | entry={line}", file=sys.stderr)
        if echo:
            print(f"[{level}] {self._component}: {message}", file=sys.stderr)

    @classmethod
    def _sanitize(cls, value: Any, key: str = "") -> Any:
        """Redact sensitive values and truncate very large strings recursively."""
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEYS):
            return "[REDACTED]"
        if isinstance(value, str):
            if len(value) > _MAX_FIELD_CHARS:
                return value[:_MAX_FIELD_CHARS] + f"... [truncated {len(value) - _MAX_FIELD_CHARS} chars]"
            return value
        if isinstance(value, Mapping):
            return {str(k): cls._sanitize(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize(v, key) for v in value[:100]] + (["... [truncated list]"] if len(value) > 100 else [])
        return value


_cache_logger: dict[str, StructuredLogger] = {}


def get_logger(component: str) -> StructuredLogger:
    """Factory function to obtain a StructuredLogger for a named component."""
    if component not in _cache_logger:
        _cache_logger[component] = StructuredLogger(component, CONFIG)
    return _cache_logger[component]
