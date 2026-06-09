# =============================================================================
# CELL 10 — Section Detector
# =============================================================================
"""
Detect annual report sections from the first 300 characters of a text block.
28+ regex patterns covering standard Indian annual report sections.
"""

import re
from typing import Optional, Set

# ── Section name constants ────────────────────────────────────────────────────
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


def detect_section(text: str, previous_section: Optional[str] = None) -> str:
    """
    Identify the annual report section from the first 300 characters of text.

    If no pattern matches, inherits the previous section. If no previous
    section is available, returns 'unknown'.

    Parameters
    ----------
    text : str
        Raw page/chunk text.
    previous_section : str, optional
        Section name from the previous chunk (for inheritance).

    Returns
    -------
    str
        Section name constant (e.g. 'management_discussion').
    """
    snippet = text[:300]
    for pattern, section_name in _SECTION_PATTERNS:
        if pattern.search(snippet):
            return section_name
    return previous_section if previous_section else SEC_UNKNOWN

# ----------------------------------------------------------------------------
# Cell 10: Section Detector
# Purpose: Classify a text block into one of 28+ named annual report sections.
# Key Classes: None (functions + constants only)
# Key Functions: detect_section(text, previous_section=None) → str
# Key Constants/Config: SEC_* constants, MGMT_SECTIONS, FINANCIAL_SECTIONS,
#   _SECTION_PATTERNS
# Imports exported: detect_section, MGMT_SECTIONS, FINANCIAL_SECTIONS, SEC_*
# Depends on: None
# Critical notes: Patterns are matched against the FIRST 300 chars only —
#   section headers appear at page tops. Section inheritance (previous_section)
#   ensures all chunks have a meaningful section label.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------


