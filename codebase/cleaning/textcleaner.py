
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from collections import defaultdict

from config import CONFIG                       # noqa: E402  (root module)
from logger import get_logger                   # noqa: E402  (root module)
from codebase.cleaning.skelton import CleanResult, TableType
from codebase.cleaning.skelton import(
    RE_HR,
    RE_IMAGE,
    RE_TABLE_ROW,
    RE_TABLE_SEP,
    RE_TOC_LINE,
    QUALITATIVE_KEYWORDS,
    FINANCIAL_KEYWORDS
)

logger = get_logger(__name__)


class TextCleaner:

    def __init__(self, company: str, year: int, doc_type: str, min_table_rows: int = 2) -> None:
        self.company        = company
        self.year           = year
        self.doc_type       = doc_type
        self.min_table_rows = min_table_rows

    #normalize endings    
    def _normalise_endings(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    #remove line artifacts
    def _remove_line_artifacts(self, text: str) -> str:
        cleaned_lines: list[str] = []

        for line in text.splitlines():
            line = line.replace("&gt;", ">").replace("&amp;", "&")

            line_stripped = line.lstrip("> ").rstrip()

            line_stripped = RE_IMAGE.sub("", line_stripped).strip()

            if not line_stripped:
                cleaned_lines.append("")
                continue

            if RE_TOC_LINE.match(line_stripped):
                logger.debug(f"[drop toc]{line_stripped[:80]}")
                continue

            if RE_HR.match(line_stripped):
                logger.debug(f"[drop hr]      {line_stripped[:40]}")
                continue

            cleaned_lines.append(line_stripped)

        return "\n".join(cleaned_lines)

    #normalize whitespace
    def _normalise_whitespace(self, text: str) -> str:
        """Collapse three or more consecutive blank lines down to two."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    #detect table
    def detect_table(self, text: str) -> bool:
        has_data_row = False
        has_sep_row  = False

        for line in text.split("\n"):
            line = line.strip()
            if RE_TABLE_SEP.match(line):
                has_sep_row = True
            elif RE_TABLE_ROW.match(line):
                has_data_row = True
            if has_data_row and has_sep_row:
                return True
        return False

    def classify_table(self, text: str) -> TableType:
        table_text = " ".join(
            line.strip()
            for line in text.split("\n")
            if RE_TABLE_ROW.match(line.strip()) or RE_TABLE_SEP.match(line.strip())
        ).lower()

        financial_score   = sum(1 for kw in FINANCIAL_KEYWORDS   if kw in table_text)
        qualitative_score = sum(1 for kw in QUALITATIVE_KEYWORDS if kw in table_text)

        if financial_score >= qualitative_score:
            return TableType.FINANCIAL
        return TableType.QUALITATIVE

    #checks count and is short
    def check_count_and_isshort(self, text: str, min_words: int = 20) -> tuple[int, bool]:
        word_count = len(text.split())
        return word_count, word_count < min_words

    #text Cleaner
    def clean(self, page_text: str, page_num: int) -> CleanResult:
        logger.info(
            f"[{self.company} {self.year}] Cleaning page {page_num} — "
            f"{len(page_text):,} chars input"
        )

        text = self._normalise_endings(page_text)
        text = self._remove_line_artifacts(text)
        text = self._normalise_whitespace(text)

        has_table  = self.detect_table(text)
        table_type = self.classify_table(text) if has_table else None
        word_count, is_short = self.check_count_and_isshort(text)

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

