from __future__ import annotations
import os
import json
from dataclasses import asdict
from enum import Enum

from config import CONFIG
from logger import get_logger
from inputvalidator import InputValidator
from healthcheck import assert_system_health

from codebase.cleaning.skelton import CleanResult
from codebase.cleaning.textcleaner import TextCleaner
from codebase.cleaning.pageintent import _tag_page_intent
from codebase.cleaning.tableinfo import TableExtractor
from codebase.cleaning.embeddingprepared import EmbeddingPrepared
from codebase.cleaning.validator import(
    _validate_filepath
)
from codebase.cleaning.exceptions import PageNotFoundError

logger = get_logger(__name__)


class PipelineRunner:

    def __init__(self, company: str = "", year: int = 0, doc_type: str = "ANNUAL_REPORT") -> None:
        self.company = InputValidator.validate_scrip(company)
        self.year = InputValidator.validate_year_int(year)
        self.doc_type = InputValidator.validate_doc_type(doc_type)

        self._cleaner = TextCleaner(self.company, self.year, self.doc_type)
        self._table_extractor = TableExtractor()
        self._embedder = EmbeddingPrepared()

    def _load_file(self, file: str) -> list[dict]:
        try:
            with open(file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {file}") from exc
        
    def _process_pages(self, pages: list[dict]) -> tuple[list[CleanResult], int]:
        results: list[CleanResult] = []
        skipped = 0

        for page in pages:
            page_number = page.get("page_number")
            if page_number is None:
                raise PageNotFoundError("Page is missing page number after validation.")

            with logger.timed("page_clean", page_num=page_number):
                result = self._cleaner.clean(page["text"], page_number)

            if result.is_short:
                skipped += 1
                continue

            result.page_intent = _tag_page_intent(result)
            results.append(result)
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
        base = os.path.splitext(input_file)[0]
        output = InputValidator.validate_output_path(f"{base}_CLEANED.json")
        json_string = self._get_cleaned_json(results)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(json_string)
        logger.info(f"[PipelineRunner] Saved cleaned JSON → {output}")
        return output

    def _save_embeddings(self, cleaned_path: str) -> str:
        base = os.path.splitext(cleaned_path)[0].replace("_CLEANED", "")
        output = InputValidator.validate_output_path(f"{base}_EMBEDDINGREADY.json")
        self._embedder.prepare_for_embedding(cleaned_path, output)
        InputValidator.validate_embedding_payload(InputValidator.load_json_file(output))
        logger.info(f"[PipelineRunner] Saved embedding JSON → {output}")
        return output

    def run(self, file: str) -> None:
        """Execute the full pipeline on input_file."""
        assert_system_health(include_llm=False)
        _validate_filepath(file)
        with logger.timed("cleaning_pipeline", company=self.company, year=self.year, input_file=input_file):
            pages = self._load_file(file)
            results, skipped = self._process_pages(pages)
            self._log_summary(pages, results, skipped)
            cleaned_path = self._save_cleaned(input_file, results)
            embedding_path = self._save_embeddings(cleaned_path)
        return None


if __name__ == "__main__":
    COMPANY = "KALYANKJIL"
    YEAR = 2023
    DOC_TYPE = "ANNUAL"
    input_file = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}",f"{COMPANY}_{DOC_TYPE}_{YEAR}.json",)
    runner = PipelineRunner(company=COMPANY, year=YEAR, doc_type="ANNUAL_REPORT")
    results = runner.run(input_file)
