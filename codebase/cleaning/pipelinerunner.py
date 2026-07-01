"""Pipeline orchestrator for the cleaning stage.

Single responsibility: sequence the steps (validate → load → clean →
tag intent → save cleaned → prepare embeddings → record to DB) and
return a PipelineOutput.

PipelineRunner owns no business logic itself — every step is delegated
to a focused collaborator injected via the constructor (DI-friendly,
mockable in tests).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum

from config import CONFIG
from inputvalidator import InputValidator
from healthcheck import assert_system_health

from codebase.cleaning.skelton import CleanResult, PipelineOutput
from codebase.cleaning.textcleaner import TextCleaner, TableExtractor, TableClassifier, ShortPageFilter
from codebase.cleaning.pageintent import tag_page_intent
from codebase.cleaning.embeddingprepared import EmbeddingPrepared
from codebase.cleaning.validator import (
    validate_input_filepath,
    validate_cleaned_json,
    validate_embedding_json,
    validate_output_path,
)
from codebase.cleaning.exceptions import PageNotFoundError
from codebase.cleaning import db


class PipelineRunner:

    def __init__(
        self,
        company:       str = "",
        year:          int = 0,
        doc_type:      str = "ANNUAL_REPORT",
        # Collaborators — injectable for testing
        cleaner:       TextCleaner      | None = None,
        extractor:     TableExtractor   | None = None,
        classifier:    TableClassifier  | None = None,
        page_filter:   ShortPageFilter  | None = None,
        embedder:      EmbeddingPrepared | None = None,
    ) -> None:
        self.company  = InputValidator.validate_scrip(company)
        self.year     = InputValidator.validate_year_int(year)
        self.doc_type = InputValidator.validate_doc_type(doc_type)

        self._cleaner    = cleaner    or TextCleaner()
        self._extractor  = extractor  or TableExtractor()
        self._classifier = classifier or TableClassifier()
        self._filter     = page_filter or ShortPageFilter()
        self._embedder   = embedder   or EmbeddingPrepared()

    # ------------------------------------------------------------------
    # Private helpers — each does exactly one thing
    # ------------------------------------------------------------------

    def _load_pages(self, file: str) -> list[dict]:
        with open(file, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _clean_page(self, raw_page: dict) -> CleanResult:
        page_num = raw_page.get("page_num")
        if page_num is None:
            raise PageNotFoundError("Page record is missing 'page_num' after validation.")

        original_text = raw_page.get("text", "")
        cleaned_text  = self._cleaner.clean(original_text)

        has_table  = self._extractor.has_table(cleaned_text)
        _, tables  = self._extractor.strip_tables(cleaned_text) if has_table else (cleaned_text, "")
        table_type = self._classifier.classify(tables) if has_table else None

        word_count, is_short = self._filter.evaluate(cleaned_text)

        return CleanResult(
            page_num      = page_num,
            original_text = original_text,
            cleaned_text  = cleaned_text,
            word_count    = word_count,
            is_short      = is_short,
            has_table     = has_table,
            table_type    = table_type,
            company       = self.company,
            year          = self.year,
            doc_type      = self.doc_type,
            raw_tables    = tables,
        )

    def _process_pages(self, pages: list[dict]) -> tuple[list[CleanResult], int]:
        results: list[CleanResult] = []
        skipped = 0
        for raw_page in pages:
            result = self._clean_page(raw_page)
            if result.is_short:
                skipped += 1
                continue
            result.page_intent = tag_page_intent(result)
            results.append(result)
        return results, skipped

    def _serialise_results(self, results: list[CleanResult]) -> str:
        def _default(obj: object) -> object:
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
        return json.dumps(
            [asdict(r) for r in results],
            indent=2,
            ensure_ascii=False,
            default=_default,
        )

    def _save_cleaned(self, file: str, results: list[CleanResult]) -> str:
        base   = os.path.splitext(file)[0]
        output = str(validate_output_path(f"{base}_CLEANED.json"))
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(self._serialise_results(results))
        return output

    def _save_embeddings(self, cleaned_path: str) -> str:
        base   = os.path.splitext(cleaned_path)[0].replace("_CLEANED", "")
        output = str(validate_output_path(f"{base}_EMBEDDINGREADY.json"))
        self._embedder.prepare_for_embedding(cleaned_path, output)
        return output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, file: str) -> PipelineOutput:
        """Execute the full pipeline on file and return a PipelineOutput."""
        assert_system_health(include_llm=False)
        validate_input_filepath(file)

        pages                  = self._load_pages(file)
        results, skipped       = self._process_pages(pages)
        cleaned_path           = self._save_cleaned(file, results)
        validate_cleaned_json(cleaned_path)
        embedding_path         = self._save_embeddings(cleaned_path)
        validate_embedding_json(embedding_path)

        filename = os.path.basename(file)
        db.insert_cleaning_record(
            filename       = filename,
            scrip          = self.company,
            year           = self.year,
            cleaned_path   = cleaned_path,
            embedding_path = embedding_path,
        )

        return PipelineOutput(
            company         = self.company,
            year            = self.year,
            doc_type        = self.doc_type,
            total_pages     = len(pages),
            pages_processed = len(results),
            pages_skipped   = skipped,
            cleaned_path    = cleaned_path,
            embedding_path  = embedding_path,
            clean_results   = results,
        )


if __name__ == "__main__":
    COMPANY  = "KALYANKJIL"
    YEAR     = 2023
    DOC_TYPE = "ANNUAL_REPORT"
    input_file = os.path.join(
        CONFIG.UPLOADS_PATH, COMPANY,
        str(YEAR), f"{COMPANY}_{YEAR}.json",
    )
    runner = PipelineRunner(company=COMPANY, year=YEAR, doc_type=DOC_TYPE)
    output = runner.run(input_file)