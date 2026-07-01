"""Data structures and static data for the cleaning pipeline.

Convention: only dataclasses, enums, frozensets, compiled regexes,
and plain constants live here. No logic, no I/O, no imports from
sibling modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from config import CONFIG


# ---------------------------------------------------------------------------
# Path / year bounds (resolved once; passed into functions as args)
# ---------------------------------------------------------------------------

MIN_YEAR: int = 1995
MAX_YEAR: int = 2026
ALLOWED_BASE: Path      = Path(CONFIG.UPLOADS_PATH).resolve()
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".json"})

# Filename convention: SCRIP_YEAR_DOCTYPE.json
# e.g. KALYANKJIL_2025_ANNUAL_REPORT.json
# Doc-type segment is case-insensitive at validation time.
ALLOWED_DOC_TYPES: frozenset[str] = frozenset({
    "ANNUAL_REPORT",
    "QUARTERLY_REPORT",
    "INVESTOR_PRESENTATION",
})

# ^[A-Za-z0-9]+_\d{4}_(ANNUAL_REPORT|QUARTERLY_REPORT|INVESTOR_PRESENTATION)\.json$
FILENAME_RE: re.Pattern = re.compile(
    r"^[A-Za-z0-9]+"
    r"_\d{4}"
    r"_(?:ANNUAL_REPORT|QUARTERLY_REPORT|INVESTOR_PRESENTATION)"
    r"\.json$",
    re.IGNORECASE,
)

# Scrip: alphanumeric only, 1–20 chars
SCRIP_RE: re.Pattern = re.compile(r"^[A-Za-z0-9]{1,20}$")

# Status values accepted by the DB CHECK constraint
ALLOWED_STATUS_VALUES: frozenset[str] = frozenset({"SUCCESS", "FAILED"})

# Max lengths that mirror fileloader/schemas.py bounds
MAX_PATH_LENGTH:   int = 1024
MAX_REASON_LENGTH: int = 1000
MAX_SCRIP_LENGTH:  int = 20


# ---------------------------------------------------------------------------
# Keyword sets used by the table classifier
# ---------------------------------------------------------------------------

FINANCIAL_KEYWORDS: frozenset[str] = frozenset({
    "revenue", "profit", "loss", "ebitda", "pbt", "pat", "eps",
    "debt", "equity", "roce", "roe", "margin", "income", "expenditure",
    "expense", "cash", "dividend", "earnings", "turnover", "assets",
    "liabilities", "borrowing", "interest", "tax", "depreciation",
    "balance sheet", "p&l", "gml", "non-gml", "mn", "million", "crore",
    "₹", "inr", "usd", "fy25", "fy24", "fy23", "fy22", "fy21",
    "%", "growth", "cagr", "return", "capital", "net worth",
})

QUALITATIVE_KEYWORDS: frozenset[str] = frozenset({
    "showroom", "store", "staff", "employee", "headcount", "branch",
    "director", "board", "committee", "member", "designation",
    "name", "appointment", "compliance", "plan", "strategy",
    "customer", "product", "region", "geography", "outlet",
    "franchise", "foco", "candere", "attendance", "meeting",
    "complaint", "si no", "sl no", "serial", "category",
})


# ---------------------------------------------------------------------------
# Compiled regexes (module-level: compiled once, reused everywhere)
# ---------------------------------------------------------------------------

RE_IMAGE:     re.Pattern = re.compile(r"!\[.*?\]\(.*?\)", re.IGNORECASE)
RE_TOC_LINE:  re.Pattern = re.compile(r"^.{3,60}\s+\.\.\.\s+\d+\s*$")
RE_HR:        re.Pattern = re.compile(r"^-{3,}\s*$")
RE_TABLE_ROW: re.Pattern = re.compile(r"^\|.+\|$")
RE_TABLE_SEP: re.Pattern = re.compile(r"^\|[-| :]+\|$")


# ---------------------------------------------------------------------------
# Section name constants + pattern registry
# ---------------------------------------------------------------------------

class SectionName:
    """String constants for every recognised section name."""

    CHAIRMAN             = "chairman_letter"
    MD_OVERVIEW          = "managing_director_overview"
    MGT_DISCUSSION       = "management_discussion"
    STRATEGY             = "strategy"
    BUSINESS_REVIEW      = "business_review"
    SEGMENT_REVIEW       = "segment_review"
    ESG                  = "esg_sustainability"
    CORPORATE_GOV        = "corporate_governance"
    RISK                 = "risk_management"
    DIRECTORS_REPORT     = "directors_report"
    FINANCIALS_STANDALONE = "financial_statements_standalone"
    FINANCIALS_CONSOL    = "financial_statements_consolidated"
    NOTES                = "notes_to_accounts"
    AUDITOR              = "auditors_report"
    BALANCE_SHEET        = "balance_sheet"
    PNL                  = "profit_and_loss"
    CASH_FLOW            = "cash_flow_statement"
    EQUITY_CHANGES       = "equity_changes"
    FIVE_YEAR            = "five_year_summary"
    TEN_YEAR             = "ten_year_summary"
    HIGHLIGHTS           = "financial_highlights"
    AWARDS               = "awards_recognition"
    BOARD                = "board_of_directors"
    KMP                  = "key_management_personnel"
    SHAREHOLDER_INFO     = "shareholder_information"
    NOTICE               = "notice_agm"
    GLOSSARY             = "glossary"
    UNKNOWN              = "unknown"

    MGMT_SECTIONS: frozenset[str] = frozenset({
        CHAIRMAN, MD_OVERVIEW, MGT_DISCUSSION,
        STRATEGY, BUSINESS_REVIEW, SEGMENT_REVIEW, DIRECTORS_REPORT,
    })

    FINANCIAL_SECTIONS: frozenset[str] = frozenset({
        FINANCIALS_STANDALONE, FINANCIALS_CONSOL, NOTES,
        BALANCE_SHEET, PNL, CASH_FLOW, EQUITY_CHANGES,
        FIVE_YEAR, TEN_YEAR, HIGHLIGHTS,
    })


# List of (compiled_regex, section_name) pairs — consumed by pageintent.py.
SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(dear\s+(share|stock)holder|chairman.{0,20}(letter|message|statement)|message from (the )?chairman)", re.I), SectionName.CHAIRMAN),
    (re.compile(r"(managing director.{0,20}(overview|message|letter)|md.{0,10}(overview|message))", re.I), SectionName.MD_OVERVIEW),
    (re.compile(r"management.{0,10}discussion.{0,10}(and\s+)?analysis|md\s*[&]\s*a\b", re.I), SectionName.MGT_DISCUSSION),
    (re.compile(r"(strategic\s+(priorities|review|outlook|pillars)|our\s+strategy|strategy\s+(overview|in\s+action))", re.I), SectionName.STRATEGY),
    (re.compile(r"(business\s+(review|performance|overview)|operating\s+review)", re.I), SectionName.BUSINESS_REVIEW),
    (re.compile(r"(segment(al)?\s+(review|performance|results)|division(al)?\s+(review|performance))", re.I), SectionName.SEGMENT_REVIEW),
    (re.compile(r"(esg|environmental.{0,10}social.{0,10}govern|sustainability\s+(report|overview|performance))", re.I), SectionName.ESG),
    (re.compile(r"(corporate\s+governance\s+report|report\s+on\s+corporate\s+governance)", re.I), SectionName.CORPORATE_GOV),
    (re.compile(r"(risk\s+management|principal\s+risks|key\s+risks)", re.I), SectionName.RISK),
    (re.compile(r"(directors['\s]?\s*report|board['\s]?\s*report)", re.I), SectionName.DIRECTORS_REPORT),
    (re.compile(r"(standalone\s+financial\s+statements?|unconsolidated\s+financial)", re.I), SectionName.FINANCIALS_STANDALONE),
    (re.compile(r"(consolidated\s+financial\s+statements?|group\s+financial\s+statements?)", re.I), SectionName.FINANCIALS_CONSOL),
    (re.compile(r"(notes?\s+(to|forming part of)\s+(the\s+)?(financial|accounts))", re.I), SectionName.NOTES),
    (re.compile(r"(auditor['\s]?s?\s+report|independent\s+auditor)", re.I), SectionName.AUDITOR),
    (re.compile(r"(balance\s+sheet|statement\s+of\s+(financial|assets))", re.I), SectionName.BALANCE_SHEET),
    (re.compile(r"(profit\s+(and|&)\s+loss|statement\s+of\s+(profit|income|operations))", re.I), SectionName.PNL),
    (re.compile(r"(cash\s+flow\s+statement|statement\s+of\s+cash\s+flows?)", re.I), SectionName.CASH_FLOW),
    (re.compile(r"(statement\s+of\s+changes?\s+in\s+equity|equity\s+reconciliation)", re.I), SectionName.EQUITY_CHANGES),
    (re.compile(r"(five.year\s+(financial\s+)?summary|5.year\s+summary)", re.I), SectionName.FIVE_YEAR),
    (re.compile(r"(ten.year\s+(financial\s+)?summary|10.year\s+summary|decade\s+at\s+a\s+glance)", re.I), SectionName.TEN_YEAR),
    (re.compile(r"(financial\s+highlights?|key\s+(financial\s+)?indicators?|performance\s+highlights?)", re.I), SectionName.HIGHLIGHTS),
    (re.compile(r"(awards?\s+(and\s+)?(recognition|accolades)|recognition\s+and\s+awards?)", re.I), SectionName.AWARDS),
    (re.compile(r"(board\s+of\s+directors|our\s+board|directors\s+profile)", re.I), SectionName.BOARD),
    (re.compile(r"(key\s+management|senior\s+(leadership|management)\s+team|executive\s+(committee|team))", re.I), SectionName.KMP),
    (re.compile(r"(shareholder\s+(information|return)|investor\s+(relations|information))", re.I), SectionName.SHAREHOLDER_INFO),
    (re.compile(r"(notice\s+of\s+(the\s+)?(annual|extraordinary)\s+general|agm\s+notice)", re.I), SectionName.NOTICE),
    (re.compile(r"(glossary|definitions|abbreviations\s+used)", re.I), SectionName.GLOSSARY),
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(Enum):
    ANNUAL_REPORT          = "ANNUAL_REPORT"
    QUARTERLY_REPORT       = "QUARTERLY_REPORT"
    INVESTOR_PRESENTATION  = "INVESTOR_PRESENTATION"


class TableType(Enum):
    FINANCIAL   = "financial"
    QUALITATIVE = "qualitative"
    COMPARISON  = "comparison"
    SCHEDULE    = "schedule"
    OTHER       = "other"


class PageIntentType(Enum):
    COVER                = "cover"
    TOC                  = "toc"
    FINANCIAL_STATEMENT  = "financial_statement"
    NOTES                = "notes"
    MANAGEMENT_DISCUSSION = "management_discussion"
    AUDIT_REPORT         = "audit_report"
    OTHER                = "other"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    """All artefacts produced while cleaning a single PDF page."""
    page_num:       int
    original_text:  str
    cleaned_text:   str
    word_count:     int
    is_short:       bool
    has_table:      bool
    company:        str                      = ""
    year:           int                      = 0
    doc_type:       str                      = ""
    table_type:     Optional[TableType]      = None
    page_intent:    list[str]                = field(default_factory=list)
    raw_tables:     str                      = ""


@dataclass
class EmbeddingReadyChunk:
    """Text chunk ready for embedding."""
    page_num:    int
    chunk_index: int
    text:        str
    tokens:      int
    metadata:    dict = field(default_factory=dict)


@dataclass
class PipelineOutput:
    """Output of the full cleaning pipeline returned by PipelineRunner.run()."""
    company:         str
    year:            int
    doc_type:        str
    total_pages:     int
    pages_processed: int
    pages_skipped:   int
    cleaned_path:    str
    embedding_path:  str
    clean_results:   list[CleanResult]       = field(default_factory=list)
    embedding_chunks: list[EmbeddingReadyChunk] = field(default_factory=list)
    errors:          list[str]               = field(default_factory=list)