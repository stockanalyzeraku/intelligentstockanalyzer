# =============================================================================
# CELL 17 — Working Memory
# =============================================================================
"""
WorkingMemory dataclass: tracks all state accumulated during one REACT cycle.
Provides deduplication, chronological ordering, and context formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from sectiondetector import SEC_FIVE_YEAR,SEC_TEN_YEAR,SEC_BALANCE_SHEET,SEC_HIGHLIGHTS,SEC_PNL,SEC_CASH_FLOW,SEC_MGT_DISCUSSION,SEC_CHAIRMAN,SEC_STRATEGY,SEC_RISK
from queryinterpreter import QueryIntent

# Section priority for context ordering (lower = higher priority)
_SECTION_PRIORITY: Dict[str, int] = {
    SEC_HIGHLIGHTS: 1,
    SEC_FIVE_YEAR: 2,
    SEC_TEN_YEAR: 3,
    SEC_PNL: 4,
    SEC_BALANCE_SHEET: 5,
    SEC_CASH_FLOW: 6,
    SEC_MGT_DISCUSSION: 7,
    SEC_CHAIRMAN: 8,
    SEC_STRATEGY: 9,
    SEC_RISK: 10,
}
_DEFAULT_SECTION_PRIORITY = 99


@dataclass
class WorkingMemory:
    """
    All state accumulated during one agent query cycle.

    Attributes
    ----------
    question : str
        Original user question.
    intent : QueryIntent or None
        Parsed query intent.
    retrieved_chunks : List[Dict]
        Deduplicated retrieved chunks (semantic + BM25).
    parent_contexts : List[Dict]
        Parent section chunks fetched for context expansion.
    identified_gaps : List[str]
        Gaps identified by gap analyser.
    tried_queries : List[str]
        Queries attempted so far (for loop avoidance).
    confidence : float
        Estimated answer confidence (0–1).
    iteration : int
        Current REACT loop iteration.
    synthesis_ready : bool
        True when gap analyser signals sufficient coverage.
    """
    question: str
    intent: Optional[QueryIntent] = None
    retrieved_chunks: List[Dict] = field(default_factory=list)
    parent_contexts: List[Dict] = field(default_factory=list)
    identified_gaps: List[str] = field(default_factory=list)
    tried_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0
    iteration: int = 0
    synthesis_ready: bool = False
    _chunk_id_set: Set[str] = field(default_factory=set, repr=False)
    _parent_id_set: Set[str] = field(default_factory=set, repr=False)

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add retrieved chunks, deduplicating by chunk ID.

        Parameters
        ----------
        chunks : List[Dict]
            Chunks from HybridRetriever.retrieve().

        Returns
        -------
        int
            Number of new (non-duplicate) chunks added.
        """
        added = 0
        for chunk in chunks:
            cid = chunk.get("id")
            if cid and cid not in self._chunk_id_set:
                self._chunk_id_set.add(cid)
                self.retrieved_chunks.append(chunk)
                added += 1
        return added

    def add_parent_context(self, parent: Dict) -> None:
        """
        Add a parent context chunk, deduplicating by ID.

        Parameters
        ----------
        parent : Dict
            Parent chunk from get_parent_context().
        """
        pid = parent.get("id")
        if pid and pid not in self._parent_id_set:
            self._parent_id_set.add(pid)
            self.parent_contexts.append(parent)

    def get_ordered_context(self) -> str:
        """
        Build a formatted context string ready for the synthesis LLM prompt.

        Ordering: oldest fiscal year first, then section priority, then page number.

        Returns
        -------
        str
            Multi-section context string with source tags.
        """
        def sort_key(chunk: Dict) -> tuple:
            meta = chunk.get("metadata", {})
            fy_int = int(meta.get("fiscal_year_int", 9999))
            section = meta.get("section", "unknown")
            priority = _SECTION_PRIORITY.get(section, _DEFAULT_SECTION_PRIORITY)
            page = int(meta.get("page_number", 0))
            return (fy_int, priority, page)

        all_chunks = list(self.retrieved_chunks)
        # Add parent contexts (if not already present)
        parent_ids_in_chunks = {c.get("id") for c in all_chunks}
        for pc in self.parent_contexts:
            if pc.get("id") not in parent_ids_in_chunks:
                all_chunks.append({"id": pc["id"], "text": pc["document"], "metadata": pc["metadata"]})

        ordered = sorted(all_chunks, key=sort_key)
        parts: List[str] = []
        for chunk in ordered:
            meta = chunk.get("metadata", {})
            source = meta.get("source_display", "Unknown Source")
            text = chunk.get("text", "")
            parts.append(f"[Source: {source}]\n{text}")

        return "\n\n---\n\n".join(parts) if parts else "No context retrieved."


# ----------------------------------------------------------------------------
# Cell 17: Working Memory
# Purpose: Track all agent state across one REACT loop iteration.
# Key Classes: WorkingMemory (dataclass)
# Key Functions:
#   WorkingMemory.add_chunks(chunks) → int
#   WorkingMemory.add_parent_context(parent) → None
#   WorkingMemory.get_ordered_context() → str
# Key Constants/Config: _SECTION_PRIORITY dict
# Imports exported: WorkingMemory
# Depends on: Cell 10 (SEC_* constants), Cell 16 (QueryIntent)
# Critical notes: WorkingMemory is instantiated fresh per ask() call.
#   get_ordered_context() produces the exact string passed to the synthesis LLM.
#   Parent contexts are appended at the end to avoid duplication.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------


