"""Shared types and data structures across all modules."""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# ─────────────────────────────────────────────────────────────────
# Enums (status, types)
# ─────────────────────────────────────────────────────────────────

class DocumentType(Enum):
    """Document types supported by the pipeline."""
    ANNUAL_REPORT = "ANNUAL_REPORT"
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"

class PDFProcessingStatus(Enum):
    """Status of PDF processing."""
    REGISTERED = "registered"
    OCR_COMPLETE = "ocr_complete"
    CLEANED = "cleaned"
    EMBEDDED = "embedded"
    VECTORIZED = "vectorized"
    FAILED = "failed"
class QueryIntentType(Enum):
    """Type of financial query."""
    METRIC_LOOKUP = "metric_lookup"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    RATIO_ANALYSIS = "ratio_analysis"

# ─────────────────────────────────────────────────────────────────
# Base Types (shared across modules)
# ─────────────────────────────────────────────────────────────────

@dataclass
class CompanyMetadata:
    """Metadata about a company."""
    company_id: int
    name: str
    symbol: str
    sector: Optional[str] = None
    exchange: Optional[str] = None
    website: Optional[str] = None

@dataclass
class DocumentMetadata:
    """Metadata about a financial document."""
    doc_id: int
    company: str
    doc_type: DocumentType
    fiscal_year: int
    file_path: str
    file_size_bytes: int
    processing_status: PDFProcessingStatus
    created_at: str
    updated_at: str

@dataclass
class FinancialDataPoint:
    """Single financial metric value."""
    company_id: int
    line_item: str
    period_label: str
    fiscal_year_end: str
    value: float
    unit: str  # e.g., "INR_CRORE", "PERCENT"
    scraped_at: str
@dataclass
class FinancialSeries:
    """Multiple financial data points for one metric."""
    company: str
    line_item: str
    unit: str
    data_points: list[FinancialDataPoint] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.data_points)

@dataclass
class CacheEntry:
    """Entry in the query response cache."""
    cache_key: str
    question: str
    company: str
    year: Optional[int]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    expires_at: Optional[str] = None