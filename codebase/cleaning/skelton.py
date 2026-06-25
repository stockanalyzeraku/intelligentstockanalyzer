"""Data structures for the cleaning pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path
from config import CONFIG
import re
from typing import Set

MIN_YEAR = 1995
MAX_YEAR = 2026
ALLOWED_BASE = Path(CONFIG.UPLOADS_PATH).resolve()
ALLOWED_EXTENSIONS = {".json"}


FINANCIAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "revenue", "profit", "loss", "ebitda", "pbt", "pat", "eps",
        "debt", "equity", "roce", "roe", "margin", "income", "expenditure",
        "expense", "cash", "dividend", "earnings", "turnover", "assets",
        "liabilities", "borrowing", "interest", "tax", "depreciation",
        "balance sheet", "p&l", "gml", "non-gml", "mn", "million", "crore",
        "₹", "inr", "usd", "fy25", "fy24", "fy23", "fy22", "fy21",
        "%", "growth", "cagr", "return", "capital", "net worth",
    }
)

QUALITATIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "showroom", "store", "staff", "employee", "headcount", "branch",
        "director", "board", "committee", "member", "designation",
        "name", "appointment", "compliance", "plan", "strategy",
        "customer", "product", "region", "geography", "outlet",
        "franchise", "foco", "candere", "attendance", "meeting",
        "complaint", "si no", "sl no", "serial", "category",
    }
)

# Mistral OCR image reference:  ![img-12.jpeg](img-12.jpeg)
RE_IMAGE = re.compile(r"!\[.*?\]\(.*?\)", re.IGNORECASE)

# Table-of-contents lines:  "Performance Highlights ... 24"
RE_TOC_LINE = re.compile(r"^.{3,60}\s+\.\.\.\s+\d+\s*$")

# Horizontal rules:  ---
RE_HR = re.compile(r"^-{3,}\s*$")
RE_TABLE_ROW = re.compile(r"^\|.+\|$")
RE_TABLE_SEP = re.compile(r"^\|[-| :]+\|$")

class DocumentType(Enum):
    """Document types supported by the pipeline."""
    ANNUAL_REPORT = "ANNUAL_REPORT"
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"

@dataclass
class SectionPattern():
    SEC_CHAIRMAN = "chairman_letter"
    SEC_MD_OVERVIEW = "managing_director_overview"
    SEC_MGT_DISCUSSION = "management_discussion"
    SEC_STRATEGY = "strategy"
    SEC_BUSINESS_REVIEW = "business_review"
    SEC_SEGMENT_REVIEW = "segment_review"
    SEC_ESG = "esg_sustainability"
    SEC_CORPORATE_GOV = "corporate_governance"
    SEC_RISK = "risk_management"
    SEC_DIRECTORS_REPORT = "directors_report"
    SEC_FINANCIALS_STANDALONE = "financial_statements_standalone"
    SEC_FINANCIALS_CONSOL = "financial_statements_consolidated"
    SEC_NOTES = "notes_to_accounts"
    SEC_AUDITOR = "auditors_report"
    SEC_BALANCE_SHEET = "balance_sheet"
    SEC_PNL = "profit_and_loss"
    SEC_CASH_FLOW = "cash_flow_statement"
    SEC_EQUITY_CHANGES = "equity_changes"
    SEC_FIVE_YEAR = "five_year_summary"
    SEC_TEN_YEAR = "ten_year_summary"
    SEC_HIGHLIGHTS = "financial_highlights"
    SEC_AWARDS = "awards_recognition"
    SEC_BOARD = "board_of_directors"
    SEC_KMP = "key_management_personnel"
    SEC_SHAREHOLDER_INFO = "shareholder_information"
    SEC_NOTICE = "notice_agm"
    SEC_GLOSSARY = "glossary"
    SEC_UNKNOWN = "unknown"

    # ── Section groupings ──────────────────────────────────────────────────────────
    MGMT_SECTIONS: Set[str] = {
        SEC_CHAIRMAN, SEC_MD_OVERVIEW, SEC_MGT_DISCUSSION, SEC_STRATEGY,
        SEC_BUSINESS_REVIEW, SEC_SEGMENT_REVIEW, SEC_DIRECTORS_REPORT,
    }
    FINANCIAL_SECTIONS: Set[str] = {
        SEC_FINANCIALS_STANDALONE, SEC_FINANCIALS_CONSOL, SEC_NOTES,
        SEC_BALANCE_SHEET, SEC_PNL, SEC_CASH_FLOW, SEC_EQUITY_CHANGES,
        SEC_FIVE_YEAR, SEC_TEN_YEAR, SEC_HIGHLIGHTS,
    }

    # ── Pattern registry: (compiled_regex, section_name) ─────────────────────────
    _SECTION_PATTERNS = [
        (re.compile(r"(dear\s+(share|stock)holder|chairman.{0,20}(letter|message|statement)|message from (the )?chairman)", re.I), SEC_CHAIRMAN),
        (re.compile(r"(managing director.{0,20}(overview|message|letter)|md.{0,10}(overview|message))", re.I), SEC_MD_OVERVIEW),
        (re.compile(r"management.{0,10}discussion.{0,10}(and\s+)?analysis|md\s*[&]\s*a\b", re.I), SEC_MGT_DISCUSSION),
        (re.compile(r"(strategic\s+(priorities|review|outlook|pillars)|our\s+strategy|strategy\s+(overview|in\s+action))", re.I), SEC_STRATEGY),
        (re.compile(r"(business\s+(review|performance|overview)|operating\s+review)", re.I), SEC_BUSINESS_REVIEW),
        (re.compile(r"(segment(al)?\s+(review|performance|results)|division(al)?\s+(review|performance))", re.I), SEC_SEGMENT_REVIEW),
        (re.compile(r"(esg|environmental.{0,10}social.{0,10}govern|sustainability\s+(report|overview|performance))", re.I), SEC_ESG),
        (re.compile(r"(corporate\s+governance\s+report|report\s+on\s+corporate\s+governance)", re.I), SEC_CORPORATE_GOV),
        (re.compile(r"(risk\s+management|principal\s+risks|key\s+risks)", re.I), SEC_RISK),
        (re.compile(r"(directors['\s]?\s*report|board['\s]?\s*report)", re.I), SEC_DIRECTORS_REPORT),
        (re.compile(r"(standalone\s+financial\s+statements?|unconsolidated\s+financial)", re.I), SEC_FINANCIALS_STANDALONE),
        (re.compile(r"(consolidated\s+financial\s+statements?|group\s+financial\s+statements?)", re.I), SEC_FINANCIALS_CONSOL),
        (re.compile(r"(notes?\s+(to|forming part of)\s+(the\s+)?(financial|accounts))", re.I), SEC_NOTES),
        (re.compile(r"(auditor['\s]?s?\s+report|independent\s+auditor)", re.I), SEC_AUDITOR),
        (re.compile(r"(balance\s+sheet|statement\s+of\s+(financial|assets))", re.I), SEC_BALANCE_SHEET),
        (re.compile(r"(profit\s+(and|&)\s+loss|statement\s+of\s+(profit|income|operations))", re.I), SEC_PNL),
        (re.compile(r"(cash\s+flow\s+statement|statement\s+of\s+cash\s+flows?)", re.I), SEC_CASH_FLOW),
        (re.compile(r"(statement\s+of\s+changes?\s+in\s+equity|equity\s+reconciliation)", re.I), SEC_EQUITY_CHANGES),
        (re.compile(r"(five.year\s+(financial\s+)?summary|5.year\s+summary)", re.I), SEC_FIVE_YEAR),
        (re.compile(r"(ten.year\s+(financial\s+)?summary|10.year\s+summary|decade\s+at\s+a\s+glance)", re.I), SEC_TEN_YEAR),
        (re.compile(r"(financial\s+highlights?|key\s+(financial\s+)?indicators?|performance\s+highlights?)", re.I), SEC_HIGHLIGHTS),
        (re.compile(r"(awards?\s+(and\s+)?(recognition|accolades)|recognition\s+and\s+awards?)", re.I), SEC_AWARDS),
        (re.compile(r"(board\s+of\s+directors|our\s+board|directors\s+profile)", re.I), SEC_BOARD),
        (re.compile(r"(key\s+management|senior\s+(leadership|management)\s+team|executive\s+(committee|team))", re.I), SEC_KMP),
        (re.compile(r"(shareholder\s+(information|return)|investor\s+(relations|information))", re.I), SEC_SHAREHOLDER_INFO),
        (re.compile(r"(notice\s+of\s+(the\s+)?(annual|extraordinary)\s+general|agm\s+notice)", re.I), SEC_NOTICE),
        (re.compile(r"(glossary|definitions|abbreviations\s+used)", re.I), SEC_GLOSSARY),
    ]


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