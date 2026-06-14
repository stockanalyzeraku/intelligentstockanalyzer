"""
cleaning/embeddingprepared.py
=============================
Chunk splitting and ChromaDB ingestion helpers.

Classes
-------
EmbeddingPrepared
    Converts cleaned page JSON into fixed-size word-based chunks suitable
    for embedding with ``all-MiniLM-L6-v2`` (max ~256 tokens ≈ 180 words),
    then persists them to a JSON file and optionally ingests into ChromaDB.

Design notes
------------
- ``store_in_chromadb`` was previously a module-level function outside the
  class, which prevented it from being used in an object-oriented pipeline.
  It is now an instance method.
- ChromaDB metadata values must be flat scalars (str / int / float / bool).
  Nested dicts and lists from the page metadata are dropped automatically.
- The chunk ID format ``page_{N}_text_chunk_{C}`` is stable across runs for
  the same input file, enabling idempotent upserts.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for root-level imports
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG       # noqa: E402
from logger import get_logger   # noqa: E402

logger = get_logger(__name__)


class EmbeddingPrepared:
    """
    Splits cleaned page text into embeddable chunks and manages ChromaDB
    ingestion.

    Parameters
    ----------
    max_words : int
        Maximum number of words per chunk.  Defaults to ``180``, which
        keeps chunks comfortably below the 256-token limit of
        ``all-MiniLM-L6-v2``.
    overlap_words : int
        Number of words shared between consecutive chunks to preserve
        cross-boundary context.  Defaults to ``30``.
    """

    MAX_WORDS:    int = 150
    OVERLAP_WORDS: int = 30

    def __init__(
        self,
        max_words:    int = MAX_WORDS,
        overlap_words: int = OVERLAP_WORDS,
    ) -> None:
        self.max_words    = max_words
        self.overlap_words = overlap_words
        logger.info(
            f"[EmbeddingPrepared] Initialised — "
            f"max_words={max_words}, overlap={overlap_words}"
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def split_text_into_chunks(
        self,
        text:      str,
        max_words: int | None = None,
        overlap:   int | None = None,
    ) -> list[str]:
        """
        Split *text* into word-based chunks with overlap.

        Strategy
        --------
        1. Split on double-newlines (paragraph boundaries).
        2. If a single paragraph exceeds *max_words*, hard-split it by
           words with *overlap* words of carry-over.
        3. Otherwise accumulate paragraphs until the buffer would exceed
           *max_words*, then flush with an overlap tail.

        Parameters
        ----------
        text      : str  — cleaned page prose (tables already stripped).
        max_words : int  — override instance default.
        overlap   : int  — override instance default.

        Returns
        -------
        list[str]
            One string per chunk; chunks may share up to *overlap* words
            with their neighbours.
        """
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

    # ------------------------------------------------------------------
    # Prepare JSON for embedding
    # ------------------------------------------------------------------

    def prepare_for_embedding(
        self,
        input_path:  str,
        output_path: str,
        max_words:   int | None = None,
        overlap:     int | None = None,
    ) -> list[dict]:
        """
        Read cleaned page JSON, split prose into chunks, and write an
        embedding-ready JSON file.

        Input format
        ------------
        A JSON array where each element is a serialised ``CleanResult``
        (produced by ``dataclasses.asdict``).

        Output format
        -------------
        A JSON array where each element is::

            {
                "id":       "page_<N>_text_chunk_<C>",
                "text":     "<chunk text>",
                "metadata": { <all CleanResult fields except clean_text
                               and raw_tables> }
            }

        Parameters
        ----------
        input_path  : str  — path to the page-intent JSON.
        output_path : str  — destination path for embedding-ready JSON.
        max_words   : int  — override chunk size.
        overlap     : int  — override overlap size.

        Returns
        -------
        list[dict]  — the same records written to *output_path*.
        """
        logger.info(f"[EmbeddingPrepared] prepare_for_embedding: {input_path}")

        with open(input_path, "r", encoding="utf-8") as fh:
            data: list[dict] = json.load(fh)

        logger.info(f"[EmbeddingPrepared] Loaded {len(data)} pages")

        records: list[dict] = []

        for page in data:
            # Convert page_intent list to comma-separated string
            intents    = page.get("page_intent", [])
            intent_str = ",".join(intents) if isinstance(intents, list) else ""

            # Strip prose and tables from metadata, keep flat scalars
            metadata = {k: v for k, v in page.items() if k not in ("clean_text", "raw_tables", "page_intent")
             and isinstance(v, (str, int, float, bool))}

            # Add intent as flat string
            metadata["page_intent"] = intent_str  # "table_of_contents,company_overview,performance_highlights"
            

            clean_text = page.get("clean_text", "")
            if not clean_text.strip():
                logger.debug(
                    f"[EmbeddingPrepared] Page {page.get('page_number')} "
                    f"has no clean_text — skipping"
                )
                continue

            text_chunks = self.split_text_into_chunks(clean_text, max_words, overlap)

            for c_idx, chunk in enumerate(text_chunks):
                records.append(
                    {
                        "id":       f"page_{page['page_number']}_text_chunk_{c_idx}",
                        "text":     chunk,
                        "metadata": metadata,
                    }
                )

        logger.info(
            f"[EmbeddingPrepared] {len(records)} chunks generated from "
            f"{len(data)} pages"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

        logger.info(f"[EmbeddingPrepared] Embedding-ready JSON written → {output_path}")
        return records

    # ------------------------------------------------------------------
    # ChromaDB ingestion
    # ------------------------------------------------------------------

    def store_in_chromadb(
        self,
        embedding_json_path: str,
        collection_name:     str = "",
        chroma_path:         str | None = None,
    ) -> Any:
        """
        Load embedding-ready chunks and upsert them into a ChromaDB
        persistent collection.

        The collection uses the project-wide ``EMBEDDER`` singleton so
        that embedding happens automatically on ``collection.add()``.

        Parameters
        ----------
        embedding_json_path : str
            Path to the JSON produced by :meth:`prepare_for_embedding`.
        collection_name : str
            ChromaDB collection name.  Defaults to ``"kalyan_annual_report"``.
        chroma_path : str | None
            Override for the ChromaDB persistence directory.
            Defaults to ``CONFIG.CHROMA_PATH``.

        Returns
        -------
        chromadb.Collection
            The collection after upsert.

        Notes
        -----
        ChromaDB metadata must be flat scalars; nested objects are silently
        dropped during preparation.
        """
        import chromadb
        from codebase.vectordb.embedder import EMBEDDER   # root-level singleton

        chroma_path = chroma_path or CONFIG.CHROMA_PATH

        logger.info(
            f"[EmbeddingPrepared] store_in_chromadb — "
            f"collection='{collection_name}', path='{chroma_path}'"
        )

        with open(embedding_json_path, "r", encoding="utf-8") as fh:
            records: list[dict] = json.load(fh)

        logger.info(f"[EmbeddingPrepared] Loaded {len(records)} chunks for ingestion")

        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=EMBEDDER,
        )

        ids:       list[str]  = []
        documents: list[str]  = []
        metadatas: list[dict] = []

        for rec in records:
            ids.append(rec["id"])
            documents.append(rec["text"])

            # Flatten metadata to ChromaDB-safe scalars
            meta: dict = {
                k: v
                for k, v in rec.get("metadata", {}).items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(meta)

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(
            f"[EmbeddingPrepared] Upserted {len(ids)} chunks into "
            f"collection '{collection_name}'"
        )
        return collection