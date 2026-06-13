"""
cleaning/textcleaner.py
=======================
Cleans raw Mistral OCR markdown output of an annual report, page by page.

Responsibilities
----------------
1. Normalise line endings (CRLF → LF).
2. Remove image tags, page-footer boilerplate, ToC lines, and horizontal
   rules line-by-line.
3. Detect and classify markdown pipe-tables as FINANCIAL or QUALITATIVE.
4. Count words and flag short pages (cover pages, dividers, OCR noise).
5. Return a populated :class:`CleanResult` for every page.

Typical usage
-------------
::

    from cleaning.textcleaner import TextCleaner
    from cleaning.tableinfo   import TableExtractor
    from cleaning.pageintent  import PageIntentTagger

    cleaner   = TextCleaner("KALYANKJIL", 2025, "ANNUAL_REPORT")
    extractor = TableExtractor()
    tagger    = PageIntentTagger()

    result = cleaner.clean(page_text, page_num)
    result.page_intent               = tagger._tag_page(result)
    result.clean_text, result.raw_tables = extractor.strip_tables(result.clean_text)

Notes
-----
- ``_remove_line_artifacts`` contained a silent bug in the original code:
  valid lines were never appended to ``cleaned_lines``, so the method
  always returned an empty string.  This has been fixed.
- ``TextCleanPatterns`` is an unused dataclass from an earlier design
  iteration; it is preserved but commented out rather than deleted.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that root-level modules
# (config, logger, etc.) can be imported when this file is executed
# directly or imported as part of the cleaning package.
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG                       # noqa: E402  (root module)
from logger import get_logger                   # noqa: E402  (root module)
from codebase.cleaning.cleanresult import CleanResult, TableType

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex constants (module-level — compiled once, reused per call)
# ---------------------------------------------------------------------------

_RE_TABLE_ROW = re.compile(r"^\|.+\|$")
_RE_TABLE_SEP = re.compile(r"^\|[-| :]+\|$")

# Keywords that strongly indicate financial data
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

# Keywords that indicate qualitative / operational data
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


# ---------------------------------------------------------------------------
# Unused legacy dataclass — preserved but not active
# ---------------------------------------------------------------------------

# @dataclass
# class TextCleanPatterns:
#     """
#     Patterns to clean data and mark data to different classes.
#     Different classes of data:
#         Qualitative Text
#         Qualitative Table - Timeline
#         Qualitative Table - Facts
#         Qualitative Table - Financial
#     """
#     raw_markdown: str
#     preceding_heading: str
#     preceding_lines: str
#     company: str
#     year: int
#     page_hint: Optional[int] = None


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    Cleans a single annual report's OCR text, one page at a time.

    Parameters
    ----------
    company : str
        BSE ticker / identifier, e.g. ``"KALYANKJIL"``.
    year : int
        Financial year end, e.g. ``2025`` for FY 2024-25.
    doc_type : str
        Document category passed through to ``CleanResult``.
    min_table_rows : int
        Pipe-tables with fewer than this many rows are treated as inline
        lists and left in the narrative.  Default ``2`` (header + at
        least one data row).
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
        logger.info(
            f"[TextCleaner] Initialised — company={company}, "
            f"year={year}, doc_type={doc_type}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

        Bug fix
        -------
        The original implementation never appended passing lines to
        ``cleaned_lines``, causing the method to always return an empty
        string.  The ``else: cleaned_lines.append(line_stripped)`` branch
        below is the fix.
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

            if _RE_PAGE_FOOTER.match(line_stripped):
                logger.debug(f"[drop footer]  {line_stripped[:80]}")
                continue

            if _RE_TOC_LINE.match(line_stripped):
                logger.debug(f"[drop toc]     {line_stripped[:80]}")
                continue

            if _RE_HR.match(line_stripped):
                logger.debug(f"[drop hr]      {line_stripped[:40]}")
                continue

            # ── Line passed all filters — keep it ──────────────────────
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

    def classify_table(self, text: str) -> TableType:
        """
        Classify the table(s) in *text* as ``FINANCIAL`` or ``QUALITATIVE``.

        Scores the table content against two keyword sets.  Financial wins
        on a tie (stricter classification favours structured data routing).

        Parameters
        ----------
        text : str
            Page text that is known to contain at least one pipe-table
            (i.e. :meth:`detect_table` returned ``True``).

        Returns
        -------
        TableType
            ``TableType.FINANCIAL`` or ``TableType.QUALITATIVE``.
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

        Short pages are typically cover pages, dividers, or OCR noise and
        should be skipped during embedding.

        Parameters
        ----------
        text      : str   — text to count.
        min_words : int   — threshold below which a page is flagged short.

        Returns
        -------
        word_count : int  — number of whitespace-separated tokens.
        is_short   : bool — ``True`` when word_count < min_words.
        """
        word_count = len(text.split())
        return word_count, word_count < min_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw_text: str, page_num: int) -> CleanResult:
        """
        Full single-page cleaning pipeline.

        Steps
        -----
        1. Normalise line endings.
        2. Remove images, footers, ToC lines, horizontal rules.
        3. Normalise whitespace.
        4. Detect and classify any pipe-table.
        5. Count words and flag short pages.

        Note: pipe-table extraction (``strip_tables``) is **not** called
        here — it is applied by the caller after intent tagging, so that
        ``PageIntentTagger`` can still inspect raw table content.

        Parameters
        ----------
        raw_text : str
            Markdown text for one page as returned by Mistral OCR.
        page_num : int
            1-based page number from the source PDF.

        Returns
        -------
        CleanResult
        """
        logger.info(
            f"[{self.company} {self.year}] Cleaning page {page_num} — "
            f"{len(raw_text):,} chars input"
        )

        text = self._normalise_endings(raw_text)
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


# ---------------------------------------------------------------------------
# Entry point — end-to-end cleaning run for KALYANKJIL ANNUAL 2025
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import os

#     # ── Input / output paths follow uploads/Company/DocType+Year/File ──
#     COMPANY   = "KALYANKJIL"
#     YEAR      = 2025
#     DOC_TYPE  = "ANNUAL"

#     base_dir    = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}")
#     input_file  = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.json")
#     intent_file = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_PAGEINTENT_{YEAR}.json")
#     embed_file  = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_EMBEDDINGREADY_{YEAR}.json")

#     logger.info(f"Loading pages from: {input_file}")
#     with open(input_file, "r", encoding="utf-8") as fh:
#         pages = json.load(fh)
#     logger.info(f"Loaded {len(pages)} pages")

#     # ── Initialise pipeline components ─────────────────────────────────
#     cleaner         = TextCleaner(COMPANY, YEAR, "ANNUAL_REPORT")
#     intent_tagger   = PageIntentTagger()
#     table_extractor = TableExtractor()

#     cleanresult_book: List[CleanResult] = []
#     skipped = 0

#     for page in pages:
#         result = cleaner.clean(page["text"], page["page_num"])

#         if result.is_short:
#             logger.debug(f"Skipping short page {result.page_number} ({result.word_count} words)")
#             skipped += 1
#             continue

#         # Intent tagging operates on text that still contains tables
#         result.page_intent = intent_tagger._tag_page(result)

#         # Strip tables from prose after intent tagging
#         result.clean_text, result.raw_tables = table_extractor.strip_tables(result.clean_text)

#         # Re-count words now that tables are removed
#         result.word_count, result.is_short = cleaner.check_count(result.clean_text)

#         cleanresult_book.append(result)

#     logger.info(
#         f"Cleaning complete — {len(cleanresult_book)} pages kept, "
#         f"{skipped} short pages skipped"
#     )

#     # ── Serialise page-intent results ───────────────────────────────────
#     with open(intent_file, "w", encoding="utf-8") as fh:
#         json.dump([asdict(r) for r in cleanresult_book], fh, indent=2, ensure_ascii=False)
#     logger.info(f"Page-intent JSON written → {intent_file}")

#     # ── Prepare chunks for embedding ────────────────────────────────────
#     embedding_preparer = EmbeddingPrepared()
#     embedding_preparer.prepare_for_embedding(intent_file, embed_file)
#     logger.info(f"Embedding-ready JSON written → {embed_file}")