"""
pipelinerunner.py
=================
Standalone end-to-end test runner for the cleaning pipeline.

Lives at project root alongside config.py.
Imports from codebase.cleaning package but has no logic of its own —
it simply wires the stages together and reports results.

Usage
-----
Run from project root:

    python pipelinerunner.py

Or import and call programmatically:

    from pipelinerunner import PipelineRunner
    runner = PipelineRunner("KALYANKJIL", 2025, "ANNUAL_REPORT")
    results = runner.run("uploads/KALYANKJIL/ANNUAL_2025/1-50.json")
"""

from __future__ import annotations
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
from dataclasses import asdict

from sklearn import base

from config import CONFIG
from logger import get_logger
from codebase.cleaning.cleanresult import CleanResult
from codebase.cleaning.textcleaner import TextCleaner
from codebase.cleaning.pageintent import PageIntentTagger
from codebase.cleaning.tableinfo import TableExtractor
from codebase.cleaning.embeddingprepared import EmbeddingPrepared
from enum import Enum


logger = get_logger(__name__)


class PipelineRunner:
    """
    Wires every cleaning stage together and runs them in order:

        Load JSON
            ↓
        Clean (remove noise, detect tables, count words)
            ↓
        Skip short pages
            ↓
        Tag page intent
            ↓
        Strip tables from prose
            ↓
        Recount words
            ↓
        Save CLEANED.json
            ↓
        Prepare embedding-ready chunks → save EMBEDDINGREADY.json

    Parameters
    ----------
    company  : str — BSE ticker, e.g. "KALYANKJIL"
    year     : int — financial year end, e.g. 2025
    doc_type : str — document category, e.g. "ANNUAL_REPORT"
    """

    def __init__(
        self,
        company:  str = "KALYANKJIL",
        year:     int = 2025,
        doc_type: str = "ANNUAL_REPORT",
    ) -> None:
        self.company  = company
        self.year     = year
        self.doc_type = doc_type

        # Initialise all pipeline components once
        self._cleaner         = TextCleaner(company, year, doc_type)
        self._intent_tagger   = PageIntentTagger()
        self._table_extractor = TableExtractor()
        self._embedder        = EmbeddingPrepared()

        logger.info(
            f"[PipelineRunner] Initialised — "
            f"company={company}, year={year}, doc_type={doc_type}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_file: str) -> list[CleanResult]:
        """
        Execute the full pipeline on *input_file*.

        Parameters
        ----------
        input_file : str
            Path to the raw OCR JSON produced by MistralAIProcessor.
            Each element must have keys: ``page_num`` (or ``page_number``)
            and ``text``.

        Returns
        -------
        list[CleanResult]
            All pages that passed cleaning (short pages excluded).

        Side effects
        ------------
        Writes two files next to *input_file*:
        - ``<name>_CLEANED.json``         — serialised CleanResult list
        - ``<name>_EMBEDDINGREADY.json``  — chunked embedding records
        """
        self._log_banner("PIPELINE STARTED", input_file)

        pages = self._load(input_file)

        results, skipped = self._process_pages(pages)

        self._log_summary(pages, results, skipped)

        cleaned_path    = self._save_cleaned(input_file, results)
        embedding_path  = self._save_embeddings(cleaned_path)

        logger.info(f"[PipelineRunner] Cleaned JSON      → {cleaned_path}")
        logger.info(f"[PipelineRunner] Embedding JSON    → {embedding_path}")
        self._log_banner("PIPELINE COMPLETE", input_file)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, input_file: str) -> list[dict]:
        """Load and return the raw OCR page list from *input_file*."""
        logger.info(f"[PipelineRunner] Loading: {input_file}")
        with open(input_file, "r", encoding="utf-8") as fh:
            pages = json.load(fh)
        logger.info(f"[PipelineRunner] {len(pages)} pages loaded")
        return pages

    def _process_pages(self, pages: list[dict]) -> tuple[list[CleanResult], int]:
        """
        Run every cleaning stage on each page.

        Returns
        -------
        results : list[CleanResult] — pages that passed all filters
        skipped : int               — count of short pages dropped
        """
        results: list[CleanResult] = []
        skipped = 0

        for page in pages:
            page_num = page.get("page_num") or page.get("page_number")

            # ── Stage 1: Clean ────────────────────────────────────────
            result = self._cleaner.clean(page["text"], page_num)

            if result.is_short:
                logger.info(
                    f"  Page {page_num:>4} — SKIPPED "
                    f"({result.word_count} words below threshold)"
                )
                skipped += 1
                continue

            # ── Stage 2: Intent tagging ───────────────────────────────
            result.page_intent = self._intent_tagger._tag_page(result)
            logger.info(f"Intents for page {page_num}: {result.page_intent}")

            # ── Stage 3: Strip tables from prose ─────────────────────
            result.clean_text, result.raw_tables = (
                self._table_extractor.strip_tables(result.clean_text)
            )

            # ── Stage 4: Recount after table removal ──────────────────
            result.word_count, result.is_short = (
                self._cleaner.check_count(result.clean_text)
            )

            results.append(result)
            logger.info(
                f"  Page {page_num:>4} — OK | "
                f"words={result.word_count:>5} | "
                f"table={str(result.has_table):<5} | "
                f"type={str(result.table_type):<25} | "
                f"intents={result.page_intent}"
            )

        return results, skipped

    def _save_cleaned(self, input_file: str, results: list[CleanResult]) -> str:
        base   = os.path.splitext(input_file)[0]
        output = f"{base}_CLEANED.json"

        def default_serialiser(obj):
            if isinstance(obj, Enum):
                return obj.value   # "financial" or "qualitative" instead of TableType.FINANCIAL
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

        with open(output, "w", encoding="utf-8") as fh:
            json.dump(
                [asdict(r) for r in results],
                fh,
                indent=2,
                ensure_ascii=False,
                default=default_serialiser,
                )
        logger.info(f"[PipelineRunner] Saved cleaned JSON → {output}")
        return output

    def _save_embeddings(self, cleaned_path: str) -> str:
        """
        Run EmbeddingPrepared on the cleaned JSON and save chunks.

        Returns the embedding-ready output path.
        """
        base   = os.path.splitext(cleaned_path)[0].replace("_CLEANED", "")
        output = f"{base}_EMBEDDINGREADY.json"
        self._embedder.prepare_for_embedding(cleaned_path, output)
        logger.info(f"[PipelineRunner] Saved embedding JSON → {output}")
        return output

    def _log_banner(self, title: str, file: str) -> None:
        logger.info("=" * 60)
        logger.info(f"  {title}")
        logger.info(f"  File   : {os.path.basename(file)}")
        logger.info(f"  Ticker : {self.company} | Year: {self.year}")
        logger.info("=" * 60)

    def _log_summary(
        self,
        pages:   list[dict],
        results: list[CleanResult],
        skipped: int,
    ) -> None:
        financial = sum(
            1 for r in results
            if str(r.table_type) == "TableType.FINANCIAL"
        )
        qualitative = sum(
            1 for r in results
            if str(r.table_type) == "TableType.QUALITATIVE"
        )
        logger.info("── SUMMARY ──────────────────────────────────────────")
        logger.info(f"  Pages loaded          : {len(pages)}")
        logger.info(f"  Pages kept            : {len(results)}")
        logger.info(f"  Pages skipped (short) : {skipped}")
        logger.info(f"  Pages with tables     : {sum(1 for r in results if r.has_table)}")
        logger.info(f"  Financial tables      : {financial}")
        logger.info(f"  Qualitative tables    : {qualitative}")
        logger.info("─────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    COMPANY  = "KALYANKJIL"
    YEAR     = 2025
    DOC_TYPE = "ANNUAL"

    input_file = os.path.join(
        CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}", "1-50.json"
    )

    runner  = PipelineRunner(company=COMPANY, year=YEAR, doc_type="ANNUAL_REPORT")
    results = runner.run(input_file)

    print(f"\nDone — {len(results)} pages cleaned")