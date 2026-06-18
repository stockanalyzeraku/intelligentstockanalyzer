"""JSON debugger for every RAG worker query."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from codebase.ragrun.config import RAGRUN_CONFIG


class RAGDebugger:
    """Persist readable JSON traces for cache, retrieval, and LLM behavior."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or RAGRUN_CONFIG.debug_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, query: str, payload: dict[str, Any]) -> str:
        trace = {
            "trace_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            **payload,
        }
        path = self.output_dir / f"{self._slug(query)}_{trace['trace_id'][:8]}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._jsonable(trace), fh, ensure_ascii=False, indent=2, default=str)
        return str(path)

    def _jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value

    @staticmethod
    def _slug(value: str, max_length: int = 70) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return (slug or "ragrun")[:max_length]
