"""
Working memory for PDF processing artifacts.

This module owns the SQLite database used to remember which PDFs have been
processed and where each pipeline artifact lives: cleaned JSON,
embedding-ready JSON, and records already stored in ChromaDB.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import logging
from codebase.agentmemory.dbstructure import (
    ALL_SCHEMA_STATEMENTS,
    CHROMA_STORES_TABLE,
    PDF_ARTIFACTS_TABLE,
    PDF_FILES_TABLE,
    PROCESSING_EVENTS_TABLE,
)

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Create and manage SQLite-backed memory for processed PDF files."""

    ARTIFACT_CLEANED = "cleaned"
    ARTIFACT_EMBEDDING_READY = "embedding_ready"
    ARTIFACT_CHROMA_STORED = "chroma_stored"

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        default_db_path = Path(__file__).resolve().parents[2] / "database" / "brain.db"
        self.db_path = Path(db_path or default_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise_database()
        logger.info(f"[WorkingMemory] Initialised — db='{self.db_path}'")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with row dictionaries and FK support."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise_database(self) -> None:
        """Create all required tables, indexes, and triggers if missing."""
        with self.connect() as conn:
            for statement in ALL_SCHEMA_STATEMENTS:
                conn.execute(statement)

    def register_pdf(
        self,
        pdf_path: str | os.PathLike[str],
        company: str | None = None,
        report_type: str | None = None,
        report_year: int | None = None,
        status: str = "registered",
    ) -> int:
        """Insert or update a PDF record and return its database id."""
        path = Path(pdf_path)
        absolute_path = str(path.resolve())
        file_size = path.stat().st_size if path.exists() else None
        file_hash = self._sha256(path) if path.exists() else None

        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {PDF_FILES_TABLE} (
                    pdf_path, file_name, company, report_type, report_year,
                    file_size_bytes, file_sha256, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    company = COALESCE(excluded.company, {PDF_FILES_TABLE}.company),
                    report_type = COALESCE(excluded.report_type, {PDF_FILES_TABLE}.report_type),
                    report_year = COALESCE(excluded.report_year, {PDF_FILES_TABLE}.report_year),
                    file_size_bytes = excluded.file_size_bytes,
                    file_sha256 = excluded.file_sha256,
                    status = excluded.status
                """,
                (absolute_path, path.name, company, report_type, report_year, file_size, file_hash, status),
            )
            row = conn.execute(
                f"SELECT id FROM {PDF_FILES_TABLE} WHERE pdf_path = ?",
                (absolute_path,),
            ).fetchone()
            pdf_id = int(row["id"])

        self.log_event(pdf_id, "pdf_registered", f"Registered PDF {path.name}")
        return pdf_id

    def mark_cleaned(self, pdf_id: int, cleaned_json_path: str | os.PathLike[str], metadata: dict[str, Any] | None = None) -> None:
        """Record the cleaned JSON artifact for a PDF."""
        self.upsert_artifact(pdf_id, self.ARTIFACT_CLEANED, cleaned_json_path, True, metadata)
        self.update_pdf_status(pdf_id, "cleaned")

    def mark_embedding_ready(self, pdf_id: int, embedding_json_path: str | os.PathLike[str], metadata: dict[str, Any] | None = None) -> None:
        """Record the embedding-ready JSON artifact for a PDF."""
        self.upsert_artifact(pdf_id, self.ARTIFACT_EMBEDDING_READY, embedding_json_path, True, metadata)
        self.update_pdf_status(pdf_id, "embedding_ready")

    def mark_chroma_stored(
        self,
        pdf_id: int,
        collection_name: str,
        chroma_path: str | os.PathLike[str] | None = None,
        stored_record_count: int = 0,
        parent_collection: str | None = None,
        child_collection: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a PDF's embedding-ready records were stored in ChromaDB."""
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {CHROMA_STORES_TABLE} (
                    pdf_id, collection_name, chroma_path, parent_collection,
                    child_collection, stored_record_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id, collection_name) DO UPDATE SET
                    chroma_path = excluded.chroma_path,
                    parent_collection = excluded.parent_collection,
                    child_collection = excluded.child_collection,
                    stored_record_count = excluded.stored_record_count,
                    stored_at = CURRENT_TIMESTAMP,
                    metadata_json = excluded.metadata_json
                """,
                (
                    pdf_id,
                    collection_name,
                    str(chroma_path) if chroma_path else None,
                    parent_collection,
                    child_collection,
                    stored_record_count,
                    self._to_json(metadata),
                ),
            )
        default_chroma_path = Path(__file__).resolve().parents[2] / "chroma_db"
        self.upsert_artifact(pdf_id, self.ARTIFACT_CHROMA_STORED, chroma_path or default_chroma_path, True, metadata)
        self.update_pdf_status(pdf_id, "chroma_stored")

    def upsert_artifact(
        self,
        pdf_id: int,
        artifact_type: str,
        artifact_path: str | os.PathLike[str],
        is_ready: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a processing artifact for a PDF."""
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {PDF_ARTIFACTS_TABLE} (pdf_id, artifact_type, artifact_path, is_ready, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id, artifact_type) DO UPDATE SET
                    artifact_path = excluded.artifact_path,
                    is_ready = excluded.is_ready,
                    metadata_json = excluded.metadata_json
                """,
                (pdf_id, artifact_type, str(Path(artifact_path)), int(is_ready), self._to_json(metadata)),
            )
        self.log_event(pdf_id, f"artifact_{artifact_type}", f"Stored {artifact_type} artifact")

    def update_pdf_status(self, pdf_id: int, status: str) -> None:
        """Update a PDF processing status."""
        with self.connect() as conn:
            conn.execute(f"UPDATE {PDF_FILES_TABLE} SET status = ? WHERE id = ?", (status, pdf_id))

    def get_pdf(self, pdf_id: int) -> dict[str, Any] | None:
        """Return one PDF record with artifact and Chroma metadata."""
        with self.connect() as conn:
            pdf = conn.execute(f"SELECT * FROM {PDF_FILES_TABLE} WHERE id = ?", (pdf_id,)).fetchone()
            if pdf is None:
                return None
            artifacts = conn.execute(f"SELECT * FROM {PDF_ARTIFACTS_TABLE} WHERE pdf_id = ?", (pdf_id,)).fetchall()
            chroma = conn.execute(f"SELECT * FROM {CHROMA_STORES_TABLE} WHERE pdf_id = ?", (pdf_id,)).fetchall()
        data = dict(pdf)
        data["artifacts"] = [dict(row) for row in artifacts]
        data["chroma_stores"] = [dict(row) for row in chroma]
        return data

    def list_pdfs(self, status: str | None = None) -> list[dict[str, Any]]:
        """List tracked PDFs, optionally filtered by processing status."""
        query = f"SELECT * FROM {PDF_FILES_TABLE}"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def log_event(
        self,
        pdf_id: int | None,
        event_type: str,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append an auditable processing event."""
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {PROCESSING_EVENTS_TABLE} (pdf_id, event_type, message, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (pdf_id, event_type, message, self._to_json(metadata)),
            )

    @staticmethod
    def _to_json(value: dict[str, Any] | None) -> str | None:
        return json.dumps(value, default=str) if value is not None else None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
