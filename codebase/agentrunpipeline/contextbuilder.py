"""Context and citation construction for retrieved financial records."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any

from codebase.agentrunpipeline.citationdebugger import Citation


class FinancialContextBuilder:
    """Turn retrieval results into LLM-ready context and source citations."""

    def build(self, records: list[dict[str, Any]], max_records: int = 6) -> tuple[str, list[Citation]]:
        """Create answer context and citations from retrieved records."""
        context_blocks: list[str] = []
        citations: list[Citation] = []

        for idx, record in enumerate(records[:max_records], start=1):
            metadata = record.get("parent_metadata") or record.get("metadata") or {}
            child_metadata = record.get("child_metadata") or {}
            snippet_source = record.get("child_text") or record.get("text") or ""
            parent_text = record.get("parent_text") or record.get("text") or ""
            snippet = self._compact(snippet_source, 700)
            source_id = f"source_{idx}"

            citations.append(
                Citation(
                    source_id=source_id,
                    parent_id=record.get("parent_id") or metadata.get("parent_id"),
                    child_id=record.get("child_id"),
                    page_number=self._metadata_value(metadata, child_metadata, "page_number", "page_num"),
                    company=self._metadata_value(metadata, child_metadata, "company"),
                    report_year=self._metadata_value(metadata, child_metadata, "report_year", "year"),
                    doc_type=self._metadata_value(metadata, child_metadata, "doc_type", "report_type"),
                    page_intent=self._metadata_value(metadata, child_metadata, "page_intent"),
                    distance=record.get("distance"),
                    snippet=snippet,
                    metadata=metadata,
                )
            )
            context_blocks.append(
                "\n".join(
                    [
                        f"[{source_id}]",
                        f"metadata={metadata}",
                        f"matched_child={snippet}",
                        f"parent_context={self._compact(parent_text, 1400)}",
                    ]
                )
            )

        return "\n\n".join(context_blocks), citations

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        compacted = " ".join(str(text).split())
        return compacted if len(compacted) <= limit else compacted[: limit - 3] + "..."

    @staticmethod
    def _metadata_value(primary: dict[str, Any], secondary: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in primary and primary[key] not in (None, ""):
                return primary[key]
            if key in secondary and secondary[key] not in (None, ""):
                return secondary[key]
        return None
