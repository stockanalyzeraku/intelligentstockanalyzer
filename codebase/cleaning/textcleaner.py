"""
Cleans raw Mistral OCR markdown output of an annual report, page by page.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from collections import defaultdict

from config import CONFIG                       # noqa: E402  (root module)
from logger import get_logger                   # noqa: E402  (root module)
from codebase.cleaning.cleanresult import CleanResult, TableType

logger = get_logger(__name__)


_RE_TABLE_ROW = re.compile(r"^\|.+\|$")
_RE_TABLE_SEP = re.compile(r"^\|[-| :]+\|$")

_FINANCIAL_KEYWORDS: frozenset[str] = frozenset(
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

_QUALITATIVE_KEYWORDS: frozenset[str] = frozenset(
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
_RE_IMAGE = re.compile(r"!\[.*?\]\(.*?\)", re.IGNORECASE)

# Kalyan-specific page footer / header boilerplate
_RE_PAGE_FOOTER = re.compile(
    r"^("
    r"Kalyan Jewellers India Limited\s*//\s*Annual Report \d{4}-\d{2}"
    r"|Corporate Overview\s*//\s*Statutory Reports\s*//\s*Financial Statements"
    r"|©\s*High-Perioders.*?Annual Report.*"
    r"|\d{3}\.\s+Business Media.*?Annual Report.*"
    r"|KalyanJewellers India Limited.*?Annual Report.*"
    r"|\d+\s*$"
    r")$",
    re.IGNORECASE,
)

# Table-of-contents lines:  "Performance Highlights ... 24"
_RE_TOC_LINE = re.compile(r"^.{3,60}\s+\.\.\.\s+\d+\s*$")

# Horizontal rules:  ---
_RE_HR = re.compile(r"^-{3,}\s*$")


class TextCleaner:
    """
    Cleans a single annual report's OCR text, one page at a time.
    """

    def __init__(
        self,
        company:        str,
        year:           int,
        doc_type:       str,
        min_table_rows: int = 2,
    ) -> None:
        self.company        = company
        self.year           = year
        self.doc_type       = doc_type
        self.min_table_rows = min_table_rows
        
        logger.info(f"[TextCleaner] Initialised — company={company}, "f"year={year}, doc_type={doc_type}")

    def _normalise_endings(self, text: str) -> str:
        """Convert CRLF and bare CR to LF."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_line_artifacts(self, text: str) -> str:
        """
        Process the text line-by-line, dropping:

        - Mistral OCR image tags (``![img.jpeg](img.jpeg)``)
        - Kalyan-specific page footer / header boilerplate
        - Table-of-contents entries (``"Heading ... 42"``)
        - Horizontal rules (``---``)
        - HTML entities (``&gt;`` → ``>``, ``&amp;`` → ``&``)

        Valid lines that pass all filters are appended to the output.

        """
        cleaned_lines: list[str] = []

        for line in text.splitlines():
            # Decode common HTML entities introduced by OCR post-processing
            line = line.replace("&gt;", ">").replace("&amp;", "&")

            # Strip leading blockquote markers (``> ``) — OCR artefact
            line_stripped = line.lstrip("> ").rstrip()

            # Remove inline image tags; result may become empty
            line_stripped = _RE_IMAGE.sub("", line_stripped).strip()

            if not line_stripped:
                cleaned_lines.append("")
                continue

            # if _RE_PAGE_FOOTER.match(line_stripped):
            #     logger.debug(f"[drop footer]{line_stripped[:80]}")
            #     continue

            if _RE_TOC_LINE.match(line_stripped):
                logger.debug(f"[drop toc]{line_stripped[:80]}")
                continue

            if _RE_HR.match(line_stripped):
                logger.debug(f"[drop hr]      {line_stripped[:40]}")
                continue

            cleaned_lines.append(line_stripped)

        return "\n".join(cleaned_lines)

    def _normalise_whitespace(self, text: str) -> str:
        """Collapse three or more consecutive blank lines down to two."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def detect_table(self, text: str) -> bool:
        """
        Return ``True`` when *text* contains at least one markdown
        pipe-table (a data row **and** a separator row).

        Parameters
        ----------
        text : str
            Raw or partially-cleaned markdown text for one page.
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

    from collections import defaultdict

    def get_header_footer_for_pages(self, pages: list[dict], page_nums: list[int], min_repeat: int = 2, boundary_lines: int = 1) -> str:
    
        def get_boundary_lines(text: str, n: int) -> dict:
            """Extract first n and last n non-empty lines from page text."""
            lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 3]
        
            header_lines = lines[:n]        # first n lines
            footer_lines = lines[-n:]       # last n lines
        
            return {"header": header_lines, "footer": footer_lines}

        # Count repetitions separately for header zone and footer zone
        header_count: dict[str, set] = defaultdict(set)
        footer_count: dict[str, set] = defaultdict(set)
        page_boundaries: dict[int, dict] = {}

        for page in pages:
            page_num = page["page_number"]
            boundaries = get_boundary_lines(page["clean_text"], boundary_lines)
            page_boundaries[page_num] = boundaries
        for page_num in [1]:    
            print(page_num)
            print(f"{page_boundaries}/n")

            for line in set(boundaries["header"]):
                header_count[line].add(page_num)

            for line in set(boundaries["footer"]):
                footer_count[line].add(page_num)

        # Only keep lines that repeat across >= min_repeat pages
        repeated_headers: set[str] = {line for line, pnums in header_count.items() if len(pnums) >= min_repeat}
        repeated_footers: set[str] = {line for line, pnums in footer_count.items() if len(pnums) >= min_repeat}

        # Build result per page
        result: dict[int, dict] = {}
        for page in pages:
            page_num = page["page_number"]
            boundaries = page_boundaries[page_num]

            header = [l for l in boundaries["header"] if l in repeated_headers]
            footer = [l for l in boundaries["footer"] if l in repeated_footers]

            result[page_num] = {
                "header": "\n".join(header) if header else "",
                "footer": "\n".join(footer) if footer else "",
            }

        # Collect for queried pages
        collected = []
        for pnum in page_nums:
            entry = result.get(pnum, {})
            header = entry.get("header", "")
            footer = entry.get("footer", "")
            if header or footer:
                collected.append(f"[Page {pnum}] Header: {header or 'BLANK'} | Footer: {footer or 'BLANK'}")

        return "\n".join(collected) if collected else ""

    def classify_table(self, text: str) -> TableType:
        """
        Classify the table(s) in *text* as ``FINANCIAL`` or ``QUALITATIVE``.
        """

        table_text = " ".join(
            line.strip()
            for line in text.split("\n")
            if _RE_TABLE_ROW.match(line.strip()) or _RE_TABLE_SEP.match(line.strip())
        ).lower()

        financial_score   = sum(1 for kw in _FINANCIAL_KEYWORDS   if kw in table_text)
        qualitative_score = sum(1 for kw in _QUALITATIVE_KEYWORDS if kw in table_text)

        # Financial wins on tie — ensures numbers are never routed to ChromaDB prose
        if financial_score >= qualitative_score:
            return TableType.FINANCIAL
        return TableType.QUALITATIVE

    def check_count(self, text: str, min_words: int = 20) -> tuple[int, bool]:
        """
        Count words in *text* and flag pages that fall below *min_words*.
        """
        word_count = len(text.split())
        return word_count, word_count < min_words


    def clean(self, page_text: str, page_num: int) -> CleanResult:
        """
        Full single-page cleaning pipeline.

        Steps
        -----
        1. Normalise line endings.
        2. Remove images, footers, ToC lines, horizontal rules.
        3. Normalise whitespace.
        4. Detect and classify any pipe-table.
        5. Count words and flag short pages.
        """
        logger.info(
            f"[{self.company} {self.year}] Cleaning page {page_num} — "
            f"{len(page_text):,} chars input"
        )

        text = self._normalise_endings(page_text)
        text = self._remove_line_artifacts(text)
        text = self._normalise_whitespace(text)

        has_table  = self.detect_table(text)
        table_type = self.classify_table(text) if has_table else None
        word_count, is_short = self.check_count(text)

        logger.info(
            f"[{self.company} {self.year}] Page {page_num} done — "
            f"{len(text):,} chars, {word_count} words, "
            f"has_table={has_table}, table_type={table_type}, "
            f"is_short={is_short}"
        )

        return CleanResult(
            page_number = page_num,
            clean_text  = text,
            has_table   = has_table,
            table_type  = table_type,
            word_count  = word_count,
            is_short    = is_short,
            doc_type    = self.doc_type,
            company     = self.company,
            year        = self.year,
        )

