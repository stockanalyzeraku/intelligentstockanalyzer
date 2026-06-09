
# =============================================================================
# CELL 13 — Storage Manager
# =============================================================================
"""
StorageManager: SQLite bookkeeping + deduplication + ChromaDB upsert.
Tracks processed files and query logs. Singleton pattern.
"""

import hashlib
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional
from logger import get_logger
from config import CONFIG
import os
from chroma import CHROMA_STORE
from hierarchychunker import Chunk

class StorageManager:
    """
    Manages SQLite persistence and coordinates ChromaDB upserts.

    Tables
    ------
    processed_files : tracks which PDFs have been fully processed.
    query_log       : records every answered user query.

    Singleton — use get_instance().
    """

    _instance: Optional["StorageManager"] = None

    def __init__(self):
        """Initialise SQLite connection and create tables."""
        self._logger = get_logger("storage_manager")
        self._db_path = CONFIG.DB_PATH
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._hash_cache: Dict[str, bool] = {}  # in-memory dedup cache
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        self._chroma = CHROMA_STORE
        self._logger.info("StorageManager ready.", db=self._db_path)

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        """Create processed_files and query_log tables if they do not exist."""
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path   TEXT    NOT NULL,
                    scrip       TEXT    NOT NULL,
                    fiscal_year TEXT    NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    is_deleted  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pf_scrip  ON processed_files(scrip);
                CREATE INDEX IF NOT EXISTS idx_pf_fy     ON processed_files(fiscal_year);

                CREATE TABLE IF NOT EXISTS query_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    question     TEXT    NOT NULL,
                    scrip        TEXT,
                    fiscal_year  TEXT,
                    answer_len   INTEGER,
                    confidence   REAL,
                    iterations   INTEGER,
                    verified     INTEGER,
                    is_deleted   INTEGER NOT NULL DEFAULT 0,
                    created_at   TEXT    NOT NULL,
                    updated_at   TEXT    NOT NULL
                );
            """)

    # ── Idempotency ────────────────────────────────────────────────────────

    def is_file_processed(self, file_path: str) -> bool:
        """
        Check whether a PDF has already been fully processed.

        Parameters
        ----------
        file_path : str
            Absolute path to the PDF.

        Returns
        -------
        bool
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM processed_files WHERE file_path = ? AND is_deleted = 0",
                (file_path,),
            )
            return cursor.fetchone() is not None

    def mark_file_processed(self, file_path: str, scrip: str, fiscal_year: str, chunk_count: int) -> None:
        """
        Record a PDF as successfully processed.

        Parameters
        ----------
        file_path : str
            Absolute path to the PDF.
        scrip : str
            Company scrip.
        fiscal_year : str
            Normalised fiscal year.
        chunk_count : int
            Number of chunks stored for this file.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO processed_files
                   (file_path, scrip, fiscal_year, chunk_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_path, scrip, fiscal_year, chunk_count, now, now),
            )
        self._logger.info(
            "File marked as processed.",
            path=os.path.basename(file_path),
            scrip=scrip,
            fy=fiscal_year,
            chunks=chunk_count,
        )

    def get_processed_files(self) -> List[Dict]:
        """
        Return all non-deleted processed file records.

        Returns
        -------
        List[Dict]
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM processed_files WHERE is_deleted = 0 ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    # ── Deduplication + upsert ────────────────────────────────────────────

    @staticmethod
    def _compute_hash(text: str) -> str:
        """
        Compute MD5 of normalised first 200 characters.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            8-character hex digest.
        """
        normalised = " ".join(text[:200].lower().split())
        return hashlib.md5(normalised.encode()).hexdigest()[:8]

    def store_chunks(self, chunks: List[Chunk]) -> int:
        """
        Deduplicate by content hash and upsert new chunks into ChromaDB.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks produced by create_chunks().

        Returns
        -------
        int
            Number of new chunks actually stored (after dedup).
        """
        if not chunks:
            return 0

        # Group by target collection
        by_collection: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            if chunk.content_hash in self._hash_cache:
                continue  # already stored this session
            self._hash_cache[chunk.content_hash] = True
            by_collection.setdefault(chunk.collection, []).append(chunk)

        stored = 0
        for col_name, col_chunks in by_collection.items():
            ids = [c.id for c in col_chunks]
            docs = [c.text for c in col_chunks]
            metas = [c.to_metadata() for c in col_chunks]
            self._chroma.upsert_batch(col_name, ids, docs, metas)
            stored += len(col_chunks)

        self._logger.info("Chunks stored.", new_chunks=stored, deduplicated=len(chunks) - stored)
        return stored

    # ── Query logging ─────────────────────────────────────────────────────

    def log_query(
        self,
        question: str,
        scrip: Optional[str],
        fiscal_year: Optional[str],
        answer_len: int,
        confidence: float,
        iterations: int,
        verified: bool,
    ) -> None:
        """
        Persist a query and its result metadata to SQLite.

        Parameters
        ----------
        question : str
        scrip : str or None
        fiscal_year : str or None
        answer_len : int
        confidence : float
        iterations : int
        verified : bool
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO query_log
                   (question, scrip, fiscal_year, answer_len, confidence,
                    iterations, verified, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question, scrip, fiscal_year, answer_len, confidence,
                 iterations, int(verified), now, now),
            )

    @classmethod
    def get_instance(cls) -> "StorageManager":
        """Return the singleton StorageManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


STORAGE_MANAGER = StorageManager.get_instance()

# ----------------------------------------------------------------------------
# Cell 13: Storage Manager
# Purpose: SQLite bookkeeping, hash-based dedup, and ChromaDB upsert coordination.
# Key Classes: StorageManager
# Key Functions:
#   StorageManager.is_file_processed(file_path) → bool
#   StorageManager.mark_file_processed(file_path, scrip, fy, chunk_count) → None
#   StorageManager.store_chunks(chunks) → int
#   StorageManager.log_query(...) → None
#   StorageManager.get_processed_files() → List[Dict]
#   StorageManager._compute_hash(text) → str
#   StorageManager.get_instance() → StorageManager
# Key Constants/Config: CONFIG.DB_PATH, CONFIG.DB_BATCH_SIZE
# Imports exported: StorageManager, STORAGE_MANAGER
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 8 (CHROMA_STORE),
#   Cell 12 (Chunk)
# Critical notes: _hash_cache is in-memory — it resets on Colab restart.
#   store_chunks() is the ONLY place chunks enter ChromaDB — do not call
#   CHROMA_STORE.upsert_batch() directly from pipeline.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------


