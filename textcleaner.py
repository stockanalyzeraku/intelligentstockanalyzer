"""
cleaner.py
----------
Cleans raw Mistral OCR markdown output of an annual report.

Responsibilities:
  1. Remove image tags, page-footer boilerplate, section-dividers, ToC lines.
  2. Preserve all section headings (markdown # / ##) so downstream
     section_detector can read them.
  3. Return a single cleaned string AND a list of raw table blocks
     (markdown pipe-tables) extracted verbatim before they are stripped
     from the narrative flow.

Usage:
    from cleaner import AnnualReportCleaner
    cleaner = AnnualReportCleaner(company="KALYANKJIL", year=2025)
    result  = cleaner.clean(raw_text)
    # result.clean_text   → narrative markdown
    # result.raw_tables   → list[RawTable]
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RawTable:
    """
    One markdown pipe-table extracted verbatim from the OCR output,
    with enough context to classify it later.
    """
    raw_markdown: str          # The pipe-table text as-is
    preceding_heading: str     # Nearest ## heading above this table
    preceding_lines: str       # Up to 3 lines of prose immediately above
    company: str
    year: int
    page_hint: Optional[int] = None   # Not always available from OCR

@dataclass
class CleanResult:
    clean_text: str
    raw_tables: List[RawTable]
    company: str
    year: int

@dataclass
class TextCleanPatterns:
    """
    Patterns to clean data and mark data to different classes.
    Different classes of data:
        Qualitative Text
        Qualitative Table - Timeline
        Qualitative Table - Facts
        Qualitative Table - Financial
    """
    raw_markdown: str          # The pipe-table text as-is
    preceding_heading: str     # Nearest ## heading above this table
    preceding_lines: str       # Up to 3 lines of prose immediately above
    company: str
    year: int
    page_hint: Optional[int] = None   # Not always available from OCR

#Table Taggers
_RE_TABLE_ROW = re.compile(r'^\|.+\|$')
_RE_TABLE_SEP = re.compile(r'^\|[-| :]+\|$')

# Keywords that strongly indicate financial data
_FINANCIAL_KEYWORDS = {
    "revenue", "profit", "loss", "ebitda", "pbt", "pat", "eps",
    "debt", "equity", "roce", "roe", "margin", "income", "expenditure",
    "expense", "cash", "dividend", "earnings", "turnover", "assets",
    "liabilities", "borrowing", "interest", "tax", "depreciation",
    "balance sheet", "p&l", "gml", "non-gml", "mn", "million", "crore",
    "₹", "inr", "usd", "fy25", "fy24", "fy23", "fy22", "fy21",
    "%", "growth", "cagr", "return", "capital", "net worth",
}

# Keywords that indicate qualitative/operational data
_QUALITATIVE_KEYWORDS = {
    "showroom", "store", "staff", "employee", "headcount", "branch",
    "director", "board", "committee", "member", "designation",
    "name", "appointment", "compliance", "plan", "strategy",
    "customer", "product", "region", "geography", "outlet",
    "franchise", "foco", "candere", "attendance", "meeting",
    "complaint", "si no", "sl no", "serial", "category",
}

# Patterns

_RE_IMAGE        = re.compile(r'!\[.*?\]\(.*?\)', re.IGNORECASE)        # Mistral OCR image references:  ![img-12.jpeg](img-12.jpeg)


_RE_PAGE_FOOTER  = re.compile(
    r'^(Kalyan Jewellers India Limited\s*//\s*Annual Report \d{4}-\d{2}'
    r'|Corporate Overview\s*//\s*Statutory Reports\s*//\s*Financial Statements'
    r'|©\s*High-Perioders.*?Annual Report.*'   # OCR artefact variant
    r'|\d{3}\.\s+Business Media.*?Annual Report.*'  # another artefact variant
    r'|KalyanJewellers India Limited.*?Annual Report.*'   # short variant
    r'|\d+\s*$'   # lone page numbers
    r')$',
    re.IGNORECASE
)       #Footer

_RE_TOC_LINE     = re.compile(r'^.{3,60}\s+\.\.\.\s+\d+\s*$')       # Table-of-contents lines:  "Performance Highlights ... 24"

_RE_HR           = re.compile(r'^-{3,}\s*$')    # Horizontal rules


_RE_TABLE_BLOCK  = re.compile(
    r'((?:\|[^\n]+\|\n?)+)',   # one or more pipe rows
    re.MULTILINE
)       # Markdown pipe-table block: starts with | and continues until a blank line




class TextCleaner:
    """
    Cleans a single annual report's OCR text.

    Parameters
    ----------
    company : str
        BSE ticker / identifier, e.g. "KALYANKJIL"
    year : int
        Financial year end, e.g. 2025 for FY2024-25
    min_table_rows : int
        Pipe-tables with fewer rows than this are treated as inline lists
        and left inside the narrative (not extracted as RawTable objects).
        Default 2 (header + at least one data row).
    """

    def __init__(self, company: str, year: int, min_table_rows: int = 2):
        self.company = company
        self.year = year
        self.min_table_rows = min_table_rows

    
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_endings(self, text: str) -> str:
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def _remove_line_artifacts(self, text: str) -> str:
        """
        Process line-by-line, dropping:
          - image tags
          - page footer / header boilerplate
          - ToC entries
          - horizontal rules
          - HTML entities (&gt; → >, &amp; → &)
        """
        cleaned_lines = []
        for line in text.splitlines():
            # HTML entities first
            line = line.replace('&gt;', '>').replace('&amp;', '&')
            # Strip leading '>' blockquote markers (OCR artefact)
            line_stripped = line.lstrip('> ').rstrip()

            # Drop image tags entirely
            line_stripped = _RE_IMAGE.sub('', line_stripped).strip()

            # Skip purely empty results
            if not line_stripped:
                cleaned_lines.append('')
                continue

            # Skip footer / header boilerplate
            if _RE_PAGE_FOOTER.match(line_stripped):
                logger.debug(f"  [drop footer] {line_stripped[:80]}")
                continue

            # Skip ToC lines
            if _RE_TOC_LINE.match(line_stripped):
                logger.debug(f"  [drop toc]    {line_stripped[:80]}")
                continue

            # Skip horizontal rules
            if _RE_HR.match(line_stripped):
                continue

            cleaned_lines.append(line_stripped)
        return '\n'.join(cleaned_lines)
    
    def detect_table(text: str) -> bool:
        """
            Detects whether the text contains a markdown table.
            A valid table must have both a data row and a separator row.
            Parameters
            ----------
            text : Raw markdown text of the page.
            Returns
            -------
            bool : True if a table is found, False otherwise.
        """
        has_data_row = False
        has_sep_row  = False

        for line in text.split("\n"):
            line = line.strip()
            if _RE_TABLE_SEP.match(line):
                has_sep_row = True
            elif _RE_TABLE_ROW.match(line):
                has_data_row = True
            if has_data_row and has_sep_row:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Function 2 — Classify table as financial or qualitative
# ─────────────────────────────────────────────────────────────────────────────

    def classify_table(text: str) -> str | None:
        """
            If the text contains a table, classifies it as 'financial' or 'qualitative'.
            Financial  — contains revenue, profit, debt, ₹, FY figures, ratios etc.
            Qualitative — contains store counts, staff, directors, plans, compliance etc.
            Parameters
            ----------
            text : Raw markdown text of the page.
            Returns
            -------
            str  : "financial" | "qualitative"
            None : if no table detected in the text
        """
        if not detect_table(text):
            return None

        # Extract only the table rows for focused matching
        table_text = " ".join(
            line.strip()
            for line in text.split("\n")
            if _RE_TABLE_ROW.match(line.strip()) or _RE_TABLE_SEP.match(line.strip())
        ).lower()

    # Score against both keyword sets
    financial_score  = sum(1 for kw in _FINANCIAL_KEYWORDS  if kw in table_text)
    qualitative_score = sum(1 for kw in _QUALITATIVE_KEYWORDS if kw in table_text)

    # Tie-break: financial wins (stricter classification)
    if financial_score >= qualitative_score:
        return "financial"
    return "qualitative"


# ─────────────────────────────────────────────────────────────────────────────
#  Function 3 — Word count check and optional clean
# ─────────────────────────────────────────────────────────────────────────────

    _RE_IMAGE     = re.compile(r'!\[.*?\]\(.*?\)', re.IGNORECASE)   # ![img](url)
    _RE_MD_HEADER = re.compile(r'^#{1,3} ', re.MULTILINE)           # # ## ###
    _RE_PIPE_ROW  = re.compile(r'^\|.*\|$',  re.MULTILINE)          # table rows
    _RE_EXTRA_NL  = re.compile(r'\n{3,}')                           # 3+ blank lines


    def check_and_clean(text: str, min_words: int = 20) -> dict:
        """
            Counts words in the text. If below min_words, cleans the text.
            Cleaning removes:
            - Image markdown tags  ![img](url)
            - Leading # from headings
            -         Table rows
            - Extra blank lines
            - Leading/trailing whitespace
            Parameters
            ----------
            text      : Raw markdown text of the page.
            min_words : Minimum word threshold (default 20).
                Pages below this are likely cover pages, dividers, or noise.
            Returns
            -------
            dict with keys:
            word_count  : int   — number of words in original text
            is_short    : bool  — True if below min_words
            clean_text  : str   — cleaned text (same as input if above threshold)
        """
        word_count = len(text.split())
        is_short   = word_count < min_words
        if is_short:
            clean = text
            clean = _RE_IMAGE.sub("", clean)          # remove ![img](url)
            clean = _RE_MD_HEADER.sub("", clean)      # remove ### headings markers
            clean = _RE_PIPE_ROW.sub("", clean)       # remove table rows
            clean = _RE_EXTRA_NL.sub("\n\n", clean)   # collapse blank lines
            clean = clean.strip()
        else:
            clean = text

        return {
            "word_count" : word_count,
            "is_short"   : is_short,
            "clean_text" : clean,
        }


    # def _extract_tables(self, text: str) -> tuple[list[RawTable], str]:
    #     """
    #     Find all markdown pipe-tables in text.
    #     For each table:
    #       - capture the nearest heading and 3 prose lines above it
    #       - store as RawTable
    #       - replace the table in text with a placeholder comment so the
    #         narrative remains coherent for section detection

    #     Returns (list_of_raw_tables, text_with_tables_removed).
    #     """
    #     raw_tables: list[RawTable] = []
    #     lines = text.split('\n')

    #     # Build an index: line_number → heading that was most recently seen
    #     current_heading = "unknown"
    #     heading_at_line: dict[int, str] = {}
    #     for i, line in enumerate(lines):
    #         if line.startswith('#'):
    #             current_heading = line.lstrip('#').strip()
    #         heading_at_line[i] = current_heading

    #     # Find table spans
    #     table_spans: list[tuple[int, int]] = []   # (start_line, end_line) inclusive
    #     i = 0
    #     while i < len(lines):
    #         if lines[i].strip().startswith('|'):
    #             start = i
    #             while i < len(lines) and lines[i].strip().startswith('|'):
    #                 i += 1
    #             end = i - 1
    #             row_count = end - start + 1
    #             if row_count >= self.min_table_rows:
    #                 table_spans.append((start, end))
    #         else:
    #             i += 1

    #     # Process spans in reverse so replacements don't shift indices
    #     for (start, end) in reversed(table_spans):
    #         table_lines = lines[start:end + 1]
    #         raw_md = '\n'.join(table_lines)

    #         # Gather context: up to 3 non-empty prose lines immediately above
    #         context_lines = []
    #         scan = start - 1
    #         while scan >= 0 and len(context_lines) < 3:
    #             l = lines[scan].strip()
    #             if l and not l.startswith('|') and not l.startswith('#'):
    #                 context_lines.insert(0, l)
    #             scan -= 1
    #         preceding_prose = ' '.join(context_lines)

    #         heading = heading_at_line.get(start, "unknown")

    #         rt = RawTable(
    #             raw_markdown=raw_md,
    #             preceding_heading=heading,
    #             preceding_lines=preceding_prose,
    #             company=self.company,
    #             year=self.year,
    #         )
    #         raw_tables.append(rt)
    #         logger.debug(f"  [table] heading='{heading}' rows={end-start+1}")

    #         # Replace table in text with a lightweight placeholder
    #         placeholder = f"[TABLE_EXTRACTED: {heading[:60]}]"
    #         lines[start:end + 1] = [placeholder]

    #     # Raw tables were collected in reverse; restore document order
    #     raw_tables.reverse()

    #     return raw_tables, '\n'.join(lines)

    def _normalise_whitespace(self, text: str) -> str:
        """Collapse 3+ consecutive blank lines to 2."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw_text: str) -> CleanResult:
        """
        Full cleaning pipeline.

        Steps
        -----
        1. Normalise line endings.
        2. Remove images, footers, ToC, horizontal rules (line-by-line).
        3. Extract pipe-tables with their surrounding context.
        4. Remove extracted tables from the narrative text.
        5. Final whitespace normalisation.

        Returns CleanResult with clean_text and raw_tables list.
        """
        logger.info(f"[{self.company} {self.year}] Starting clean — "
                    f"{len(raw_text):,} chars input")

        text = self._normalise_endings(raw_text)
        text = self._remove_line_artifacts(text)
        raw_tables, text = self._extract_tables(text)
        text = self._normalise_whitespace(text)

        logger.info(f"[{self.company} {self.year}] Cleaning done — "
                    f"{len(text):,} chars output, "
                    f"{len(raw_tables)} tables extracted")

        return CleanResult(
            clean_text=text,
            raw_tables=raw_tables,
            company=self.company,
            year=self.year,
        )
