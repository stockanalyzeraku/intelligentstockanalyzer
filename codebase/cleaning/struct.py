"""Data structures for the cleaning pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from codebase.common.types import DocumentType

class PageIntentType(Enum):
    """Classification of page content."""
    COVER = "cover"
    TOC = "toc"
    FINANCIAL_STATEMENT = "financial_statement"
    NOTES = "notes"
    MANAGEMENT_DISCUSSION = "management_discussion"
    AUDIT_REPORT = "audit_report"
    OTHER = "other"

class TableType(Enum):
    """Type of table detected on page."""
    FINANCIAL = "financial"
    COMPARISON = "comparison"
    SCHEDULE = "schedule"
    OTHER = "other"
    QUALITATIVE = "qualitative"

@dataclass
class TableInfo:
    """Information about a detected table."""
    page_num: int
    table_type: TableType
    row_count: int
    col_count: int
    extracted_text: Optional[str] = None

@dataclass
class CleanResult:
    """Result of cleaning a single page."""
    page_num: int
    original_text: str
    cleaned_text: str
    word_count: int
    is_short: bool  # True if below minimum word threshold
    has_table: bool
    table_type: Optional[TableType] = None
    page_intent: Optional[list[PageIntentType]] = field(default_factory=list)
    table_info: Optional[TableInfo] = None
    doc_type:Optional[DocumentType] = None
    raw_tables:str = ""
    company:str = ""
    year:int = 0

@dataclass
class EmbeddingReadyChunk:
    """Text chunk ready for embedding."""
    page_num: int
    chunk_index: int
    text: str
    tokens: int
    metadata: dict = field(default_factory=dict)

@dataclass
class PipelineOutput:
    """Output of entire cleaning pipeline."""
    company: str
    year: int
    doc_type: str
    total_pages: int
    pages_processed: int
    pages_skipped: int
    clean_results: list[CleanResult] = field(default_factory=list)
    embedding_chunks: list[EmbeddingReadyChunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)