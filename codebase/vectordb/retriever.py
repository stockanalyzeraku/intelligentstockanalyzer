"""Parent/child aware retrieval, built on top of ChromaRecordStore.

This is the one piece of the module that's genuinely domain-specific
(it knows what a "parent section" and "child chunk" are). Keeping it
separate from ChromaRecordStore means the generic store stays generic,
and this class stays free of connection/upsert concerns.
"""

from __future__ import annotations

from typing import Any

from codebase.vectordb.skelton import ChildMatch, RetrievedItem
from codebase.vectordb.store import ChromaRecordStore


class ParentChildRetriever:
    """Searches child chunks, then expands each match to its parent section."""

    def __init__(self, store: ChromaRecordStore, parent_collection: str, child_collection: str):
        self._store = store
        self._parent_collection = parent_collection
        self._child_collection = child_collection

    def retrieve(
        self,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        raw = self._store.query(self._child_collection, query_texts, n_results, where)

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        best_children: dict[str, ChildMatch] = {}
        parent_order: list[str] = []

        for child_id, doc, meta, dist in zip(ids, docs, metas, dists):
            parent_id = (meta or {}).get("parent_id")
            if not parent_id:
                continue
            existing = best_children.get(parent_id)
            if existing is None or dist < existing.distance:
                best_children[parent_id] = ChildMatch(
                    child_id=child_id, child_text=doc, child_metadata=meta or {}, distance=dist
                )
            if parent_id not in parent_order:
                parent_order.append(parent_id)

        parents = {
            record.id: record
            for record in self._store.get_many_by_ids(self._parent_collection, parent_order)
        }

        merged: list[RetrievedItem] = []
        for parent_id in parent_order:
            parent = parents.get(parent_id)
            child = best_children.get(parent_id)
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
        return merged