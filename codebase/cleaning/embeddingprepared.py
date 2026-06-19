"""
Parent-child record preparation and ChromaDB ingestion helpers.

Classes
-------
EmbeddingPrepared
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CONFIG       
from logger import get_logger   

logger = get_logger(__name__)


class EmbeddingPrepared:

    MAX_WORDS:    int = 150
    OVERLAP_WORDS: int = 30

    def __init__(self, max_words:int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS) -> None:
        self.max_words    = max_words
        self.overlap_words = overlap_words
        logger.info(f"[EmbeddingPrepared] Initialised — "f"max_words={max_words}, overlap={overlap_words}")


    def split_text_into_chunks(self,text:str,max_words: int | None = None,overlap:int | None = None) -> list[str]:
    
        max_words = max_words if max_words is not None else self.max_words
        overlap   = overlap   if overlap   is not None else self.overlap_words

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks:         list[str]  = []
        current_words:  list[str]  = []

        for para in paragraphs:
            para_words = para.split()

            # ── Paragraph too long → hard-split ────────────────────────
            if len(para_words) > max_words:
                # Flush current buffer first
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []

                start = 0
                while start < len(para_words):
                    end = start + max_words
                    chunks.append(" ".join(para_words[start:end]))
                    next_start = end - overlap
                    start = next_start if next_start > start else end
                continue

            # ── Paragraph fits — try to accumulate ─────────────────────
            if len(current_words) + len(para_words) > max_words:
                if current_words:
                    chunks.append(" ".join(current_words))
                    overlap_tail = current_words[-overlap:] if overlap < len(current_words) else current_words[:]
                    current_words = overlap_tail + para_words
                else:
                    current_words = para_words
            else:
                current_words.extend(para_words)

        # Flush remaining words
        if current_words:
            chunks.append(" ".join(current_words))

        logger.debug(f"[EmbeddingPrepared] split_text_into_chunks → {len(chunks)} chunks")
        return chunks

    def _build_metadata(self, page: dict, record_type: str, parent_id: str, child_index: int | None = None, child_count: int | None = None) -> dict:

        intents = page.get("page_intent", [])
        intent_str = ",".join(intents) if isinstance(intents, list) else ""

        metadata = {
            k: v
            for k, v in page.items()
            if k not in ("clean_text", "raw_tables", "page_intent")
            and isinstance(v, (str, int, float, bool))
        }
        metadata["page_intent"] = intent_str
        metadata["record_type"] = record_type
        metadata["parent_id"] = parent_id
        if child_index is not None:
            metadata["child_index"] = child_index
        if child_count is not None:
            metadata["child_count"] = child_count
        return metadata


    def prepare_for_embedding(self, input_path:  str, output_path: str, max_words:   int | None = None, overlap:     int | None = None) -> dict[str, list[dict]]:
        """
        Read cleaned page JSON, build page parents plus child records, and
        write an embedding-ready JSON file.

        """
        logger.info(f"[EmbeddingPrepared] prepare_for_embedding: {input_path}")

        with open(input_path, "r", encoding="utf-8") as fh:
            data: list[dict] = json.load(fh)

        logger.info(f"[EmbeddingPrepared] Loaded {len(data)} pages")

        bundle: dict[str, list[dict]] = {"parents": [], "children": []}

        for page in data:
            clean_text = page.get("clean_text", "")
            if not clean_text.strip():
                logger.debug(
                    f"[EmbeddingPrepared] Page {page.get('page_number')} "
                    f"has no clean_text — skipping"
                )
                continue

            page_number = page.get("page_number") or page.get("page_num")
            if page_number is None:
                logger.debug("[EmbeddingPrepared] Page without page_number — skipping")
                continue

            parent_id = f"page_{page_number}_parent"
            child_chunks = self.split_text_into_chunks(clean_text, max_words, overlap)
            child_count = len(child_chunks)

            bundle["parents"].append(
                {
                    "id": parent_id,
                    "text": clean_text,
                    "metadata": self._build_metadata(
                        page=page,
                        record_type="parent",
                        parent_id=parent_id,
                        child_count=child_count,
                    ),
                }
            )

            for c_idx, chunk in enumerate(child_chunks):
                bundle["children"].append(
                    {
                        "id":       f"{parent_id}_child_{c_idx}",
                        "text":     chunk,
                        "metadata": self._build_metadata(
                            page=page,
                            record_type="child",
                            parent_id=parent_id,
                            child_index=c_idx,
                            child_count=child_count,
                        ),
                    }
                )

        logger.info(
            f"[EmbeddingPrepared] {len(bundle['parents'])} parents and {len(bundle['children'])} children generated from "
            f"{len(data)} pages"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)

        logger.info(f"[EmbeddingPrepared] Embedding-ready JSON written → {output_path}")
        return bundle

    
    # def store_in_chromadb(
    #     self,
    #     embedding_json_path: str,
    #     collection_name:     str = "",
    #     chroma_path:         str | None = None,
    # ) -> Any:
    #     """
    #     Load embedding-ready parent/child records and upsert them into the
    #     ChromaDB store.
    #     """

    #     from codebase.vectordb.chromastore import CHROMA_STORE

    #     chroma_path = chroma_path or CONFIG.CHROMA_PATH
    #     logger.info(
    #         f"[EmbeddingPrepared] store_in_chromadb — collection='{collection_name}', path='{chroma_path}'"
    #     )

    #     return CHROMA_STORE.store_in_chromadb(
    #         embedding_json_path=embedding_json_path,
    #         collection_name=collection_name,
    #         chroma_path=chroma_path,
    #     )