"""Parent/child aware retrieval, built on top of ChromaRecordStore.

This is the one piece of the module that is genuinely domain-specific:
it knows what a parent section and a child chunk are. Keeping it
separate from ChromaRecordStore means the generic store stays generic,
and this class stays free of connection/upsert concerns.
"""

from __future__ import annotations

from typing import Any

from logger import StructuredLogger

from codebase.vectordb.skelton import ChildMatch, RetrievedItem
from codebase.vectordb.store import ChromaRecordStore


class ParentChildRetriever:
    """Searches child chunks, then expands each match to its parent section."""

    def __init__(
        self,
        store: ChromaRecordStore,
        parent_collection: str,
        child_collection: str,
    ):
        self._store = store
        self._parent_collection = parent_collection
        self._child_collection = child_collection

    def retrieve(
        self,
        query_texts: list[str],
        n_results: int,
        logger: StructuredLogger,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:

        logger.event(
            f"{self._child_collection} : Child search started — "
            f"{len(query_texts)} query text(s), top_k={n_results}",
            step="retrieve_children", stage="start",
            child_collection=self._child_collection,
            query_count=len(query_texts), n_results=n_results,
        )

        raw = self._store.query(
            self._child_collection, query_texts, n_results, logger, where
        )

        ids   = raw.get("ids",       [[]])[0]
        docs  = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        logger.event(
            f"{self._child_collection} : Child search returned {len(ids)} result(s)",
            step="retrieve_children", outcome="passed",
            child_collection=self._child_collection, returned=len(ids),
        )

        # --- deduplicate: keep only the closest child per parent --------
        best_children: dict[str, ChildMatch] = {}
        parent_order: list[str] = []

        for child_id, doc, meta, dist in zip(ids, docs, metas, dists):
            parent_id = (meta or {}).get("parent_id")
            if not parent_id:
                continue
            existing = best_children.get(parent_id)
            if existing is None or dist < existing.distance:
                best_children[parent_id] = ChildMatch(
                    child_id=child_id,
                    child_text=doc,
                    child_metadata=meta or {},
                    distance=dist,
                )
            if parent_id not in parent_order:
                parent_order.append(parent_id)

        logger.event(
            f"{self._child_collection} : Deduplication complete — "
            f"{len(parent_order)} unique parent(s) to expand",
            step="deduplicate_children", outcome="passed",
            unique_parents=len(parent_order),
        )

        # --- expand to parents -----------------------------------------
        logger.event(
            f"{self._parent_collection} : Parent expansion started — "
            f"fetching {len(parent_order)} parent(s)",
            step="expand_parents", stage="start",
            parent_collection=self._parent_collection,
            parent_count=len(parent_order),
        )

        parent_records = self._store.get_many_by_ids(
            self._parent_collection, parent_order, logger
        )
        parents = {record.id: record for record in parent_records}

        logger.event(
            f"{self._parent_collection} : Parent expansion complete — "
            f"{len(parents)}/{len(parent_order)} parent(s) resolved",
            step="expand_parents", outcome="passed",
            parent_collection=self._parent_collection,
            requested=len(parent_order), resolved=len(parents),
        )

        # --- merge -------------------------------------------------------
        merged: list[RetrievedItem] = []
        for parent_id in parent_order:
            parent = parents.get(parent_id)
            child  = best_children.get(parent_id)
            if not parent or not child:
                continue
            merged.append(
                RetrievedItem(
                    id=parent.id,
                    text=parent.text,
                    metadata=parent.metadata,
                    parent_id=parent.id,
                    parent_text=parent.text,
                    parent_metadata=parent.metadata,
                    child_id=child.child_id,
                    child_text=child.child_text,
                    child_metadata=child.child_metadata,
                    distance=child.distance,
                )
            )

        logger.event(
            f"Retrieval complete — {len(merged)} merged item(s) built",
            step="merge_results", outcome="passed",
            merged_count=len(merged),
        )
        return merged