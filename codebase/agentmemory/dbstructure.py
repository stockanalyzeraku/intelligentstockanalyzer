"""
SQLite schema for persistent PDF processing memory.

The tables in this module track every PDF processed by the pipeline and the
file artifacts created at each stage: cleaned JSON, embedding-ready JSON, and
ChromaDB ingestion metadata.
"""

from __future__ import annotations

PDF_FILES_TABLE = "pdf_files"
PDF_ARTIFACTS_TABLE = "pdf_artifacts"
CHROMA_STORES_TABLE = "chroma_stores"
PROCESSING_EVENTS_TABLE = "processing_events"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS {PDF_FILES_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_path TEXT NOT NULL UNIQUE,
        file_name TEXT NOT NULL,
        company TEXT,
        report_type TEXT,
        report_year INTEGER,
        file_size_bytes INTEGER,
        file_sha256 TEXT,
        status TEXT NOT NULL DEFAULT 'registered',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {PDF_ARTIFACTS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER NOT NULL,
        artifact_type TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        is_ready INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(pdf_id, artifact_type),
        FOREIGN KEY(pdf_id) REFERENCES {PDF_FILES_TABLE}(id) ON DELETE CASCADE
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {CHROMA_STORES_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER NOT NULL,
        collection_name TEXT NOT NULL,
        chroma_path TEXT,
        parent_collection TEXT,
        child_collection TEXT,
        stored_record_count INTEGER NOT NULL DEFAULT 0,
        stored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        UNIQUE(pdf_id, collection_name),
        FOREIGN KEY(pdf_id) REFERENCES {PDF_FILES_TABLE}(id) ON DELETE CASCADE
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {PROCESSING_EVENTS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER,
        event_type TEXT NOT NULL,
        message TEXT,
        metadata_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(pdf_id) REFERENCES {PDF_FILES_TABLE}(id) ON DELETE SET NULL
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_pdf_files_status ON {PDF_FILES_TABLE}(status)",
    f"CREATE INDEX IF NOT EXISTS idx_pdf_files_company_year ON {PDF_FILES_TABLE}(company, report_year)",
    f"CREATE INDEX IF NOT EXISTS idx_pdf_artifacts_type ON {PDF_ARTIFACTS_TABLE}(artifact_type)",
    f"CREATE INDEX IF NOT EXISTS idx_processing_events_pdf ON {PROCESSING_EVENTS_TABLE}(pdf_id)",
)

UPDATE_TIMESTAMP_TRIGGER_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_{PDF_FILES_TABLE}_updated_at
    AFTER UPDATE ON {PDF_FILES_TABLE}
    FOR EACH ROW
    BEGIN
        UPDATE {PDF_FILES_TABLE} SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_{PDF_ARTIFACTS_TABLE}_updated_at
    AFTER UPDATE ON {PDF_ARTIFACTS_TABLE}
    FOR EACH ROW
    BEGIN
        UPDATE {PDF_ARTIFACTS_TABLE} SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END
    """,
)

ALL_SCHEMA_STATEMENTS: tuple[str, ...] = SCHEMA_STATEMENTS + UPDATE_TIMESTAMP_TRIGGER_STATEMENTS
