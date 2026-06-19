from __future__ import annotations
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
from dataclasses import asdict

from config import CONFIG
from logger import get_logger
from inputvalidator import InputValidator
from healthcheck import assert_system_health
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

    """

    def __init__(self, company:str = "", year:int = 0, doc_type: str = "ANNUAL_REPORT",) -> None:
        
        self.company  = InputValidator.validate_scrip(company)
        self.year     = InputValidator.validate_year_int(year)
        self.doc_type = InputValidator.validate_doc_type(doc_type)

        self._cleaner         = TextCleaner(self.company, self.year, self.doc_type)
        self._intent_tagger   = PageIntentTagger()
        self._table_extractor = TableExtractor()
        self._embedder        = EmbeddingPrepared()

        logger.info(
            f"[PipelineRunner] Initialised — "
            f"company={self.company}, year={self.year}, doc_type={self.doc_type}"
        )



    def _load(self, input_file: str) -> list[dict]:
        """Load and return the raw OCR page list from *input_file*."""        
        logger.info(f"[PipelineRunner] Loading: {input_file}")
        
        with open(input_file, "r", encoding="utf-8") as fh:
            pages = json.load(fh)
        pages = InputValidator.load_json_file(input_file)
#        pages = InputValidator.validate_raw_ocr_pages(pages)
        
        logger.info(f"[PipelineRunner] {len(pages)} pages loaded", event="raw_json_loaded", page_count=len(pages))
        return pages

    def _process_pages(self, pages: list[dict]) -> tuple[list[CleanResult], int]:
        """
        Run every cleaning stage on each page.

        """
        results: list[CleanResult] = []
        skipped = 0

        for page in pages:
            page_num = page.get("page_num") or page.get("page_number")

            if page_num is None:
                raise ValueError("Page is missing page number after validation.")

            # ── Stage 1: Clean ────────────────────────────────────────
            with logger.timed("page_clean", page_num=page_num):
                result = self._cleaner.clean(page["text"], page_num)

            if result.is_short:
                logger.process_event("page_skipped_short", "cleaning", status="skipped", page_num=page_num, word_count=result.word_count)
                skipped += 1
                continue

            # ── Stage 2: Intent tagging ───────────────────────────────
            result.page_intent = self._intent_tagger._tag_page(result)
            logger.process_event("page_intent_completed", "cleaning", page_num=page_num, intents=result.page_intent)

            # # ── Stage 3: Strip tables from prose ─────────────────────
            # result.clean_text, result.raw_tables = (
            #     self._table_extractor.strip_tables(result.clean_text)
            # )

            # ── Stage 4: Recount after table removal ──────────────────
            # result.word_count, result.is_short = self._cleaner.check_count(result.clean_text)
            # if result.is_short:
            #     logger.process_event("page_skipped_after_table removal_reason_short", "cleaning", status="skipped", page_num=page_num, word_count=result.word_count)
            #     skipped += 1
            #     continue

            results.append(result)
            logger.process_event("page_cleaned", "cleaning", page_num=page_num, word_count=result.word_count, has_table=result.has_table, table_type=str(result.table_type), intents=result.page_intent)

        return results, skipped
    
    def _get_cleaned_json(self, results: list[CleanResult]) -> str:
    
        def default_serialiser(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

        return json.dumps(
            [asdict(r) for r in results],
            indent=2,
            ensure_ascii=False,
            default=default_serialiser,
        )


    def _save_cleaned(self, input_file: str, results: list[CleanResult]) -> str:
    
        base   = os.path.splitext(input_file)[0]
        output = f"{base}_CLEANED.json"
        output = InputValidator.validate_output_path(output)

        json_string = self._get_cleaned_json(results)

        with open(output, "w", encoding="utf-8") as fh:
            fh.write(json_string)

        logger.info(f"[PipelineRunner] Saved cleaned JSON → {output}")
        return output

    # def _save_cleaned(self, input_file: str, results: list[CleanResult]) -> str:
    #     base   = os.path.splitext(input_file)[0]
    #     output = f"{base}_CLEANED.json"

    #     def default_serialiser(obj):
    #         if isinstance(obj, Enum):
    #             return obj.value   # "financial" or "qualitative" instead of TableType.FINANCIAL
    #         raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

    #     output = InputValidator.validate_output_path(output)
    #     with open(output, "w", encoding="utf-8") as fh:
    #         json.dump(
    #             [asdict(r) for r in results],
    #             fh,
    #             indent=2,
    #             ensure_ascii=False,
    #             default=default_serialiser,
    #             )
    #     logger.info(f"[PipelineRunner] Saved cleaned JSON → {output}")
    #     return output

    def _save_embeddings(self, cleaned_path: str) -> str:
        """
        Run EmbeddingPrepared on the cleaned JSON and save chunks.

        Returns the embedding-ready output path.
        """
        base   = os.path.splitext(cleaned_path)[0].replace("_CLEANED", "")
        output = f"{base}_EMBEDDINGREADY.json"
        output = InputValidator.validate_output_path(output)
        self._embedder.prepare_for_embedding(cleaned_path, output)
        InputValidator.validate_embedding_payload(InputValidator.load_json_file(output))
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

    def run(self, input_file: str) -> list[CleanResult]:
        """
        Execute the full pipeline on *input_file*.
        """

        assert_system_health(include_llm=False)
        input_file = InputValidator.validate_json_path(input_file, must_exist=True)
        self._log_banner("PIPELINE STARTED", input_file)
    
        logger.process_event("cleaning_pipeline_started", "cleaning", company=self.company, year=self.year, input_file=input_file)

        with logger.timed("cleaning_pipeline", company=self.company, year=self.year, input_file=input_file):
            
            pages = self._load(input_file)

            results, skipped = self._process_pages(pages)

            self._log_summary(pages, results, skipped)

            #-------Remove Header and Footer------
            # cleaned_json = self._get_cleaned_json(results)
            # cleaned_json = json.loads(cleaned_json)
            # header_results = self._cleaner.get_header_footer_for_pages(pages=cleaned_json, page_nums=[p["page_number"] for p in cleaned_json], min_repeat=30)
            # print(cleaned_json)
            # print(f"Header:Footer = {header_results}")
            
            cleaned_path    = self._save_cleaned(input_file, results)
            embedding_path  = self._save_embeddings(cleaned_path)

        logger.info(f"[PipelineRunner] Cleaned JSON      → {cleaned_path}")
        logger.info(f"[PipelineRunner] Embedding JSON    → {embedding_path}")
        logger.process_event("cleaning_pipeline_completed", "cleaning", company=self.company, year=self.year, cleaned_path=cleaned_path, embedding_path=embedding_path)
        self._log_banner("PIPELINE COMPLETE", input_file)

        return results


if __name__ == "__main__":

    COMPANY  = "KALYANKJIL"
    YEAR     = 2023
    DOC_TYPE = "ANNUAL"

    input_file = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}", "KALYANKJIL_ANNUAL_2023.json")        
    runner  = PipelineRunner(company=COMPANY, year=YEAR, doc_type="ANNUAL_REPORT")
    results = runner.run(input_file)

    print(f"\nDone — {len(results)} pages cleaned")