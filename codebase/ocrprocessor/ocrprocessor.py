"""
ocrprocessor.py — OCRProcessor orchestrator.

Coordinates the full pipeline:
    validate path → render pages → call OCR → validate text → save JSON

No database awareness lives here. fileloader owns record creation,
vectordb owns marking those records' ocr_status once the full
OCR → clean → embed → store pipeline completes.

The OCR client is injected via the constructor, so adding a second provider
requires no changes here — only the caller changes.

    # Default — uses Mistral
    processor = OCRProcessor()

    # Inject any client with a process_page(image_bytes) -> str method
    processor = OCRProcessor(client=GoogleOcrClient(...))
    processor = OCRProcessor(client=AWSTextractClient(...))
    processor = OCRProcessor(client=MockOcrClient())   # in tests
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import CONFIG
from logger import get_logger, StructuredLogger
from healthcheck import assert_system_health

from codebase.ocrprocessor.skelton import PageContent
from codebase.ocrprocessor.validator import (
    _validate_ocr_text,
    _validate_filepath,
    _validate_output_path,
)
from codebase.ocrprocessor.mistralclient import MistralClient
from codebase.ocrprocessor.pdfrenderer import PDFRenderer
from codebase.ocrprocessor.exceptions import FilenameValidationError
from codebase.ocrprocessor.db import mark_ocr_success, mark_ocr_failed


class OCRProcessor:
    """
    Orchestrates the full PDF → OCR → JSON pipeline.

    Args:
        client:  Any object with a process_page(image_bytes: bytes) -> str
                 method. Defaults to MistralClient if not provided.
        api_key: Mistral API key — only used when client is not supplied.
    """

    def __init__(
        self,
        client=None,
        api_key: Optional[str] = None,
    ) -> None:
        self.log = get_logger(__name__)

        if client is None:
            # Default provider. Pass a different client to use another OCR service.
            resolved_key = api_key or CONFIG.MISTRAL_API_KEY
            client = MistralClient(
                api_key=resolved_key,
                model=CONFIG.MISTRAL_MODEL_OCR,
            )

        self._client   = client
        self._renderer = PDFRenderer()

    def _process_pages(self, pdf_path: str) -> str:
        """Render every page, call OCR, validate text, collect results."""
        try:
            _validate_filepath(pdf_path)
        except:
            raise FilenameValidationError("filename", "filename is not valid")

        pages: list[PageContent] = []          # local — never carried between runs

        for page_number, image_bytes in self._renderer.render_pages(pdf_path):
            raw_text   = self._client.process_page(image_bytes)
            clean_text = _validate_ocr_text(raw_text)

            if clean_text:                     # blank pages are skipped silently
                pages.append(PageContent(page_number=page_number, text=clean_text))
                self.log.info(f"Page {page_number} — {len(clean_text):,} chars")

        output_path = os.path.splitext(pdf_path)[0] + ".json"
        return self.save_pages_to_json(pages, output_path)

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> str:
        """Validate the output path and write pages to a JSON file."""
        resolved = _validate_output_path(output_path)
        parent = Path(output_path).parent
        if str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(pc) for pc in pages]
        with open(resolved, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return str(resolved)

    def run(self, pdf_path: str, logger: StructuredLogger) -> str:
        """
        Run the full OCR pipeline on a single PDF.

        On completion, marks the processed_files row's ocr_status via
        codebase.ocrprocessor.db (mark_ocr_success / mark_ocr_failed) —
        identity is parsed from pdf_path's own filename.

            SUCCESS — pipeline completed and wrote the output JSON.
            FAILED  — any exception was raised; ocr_reason carries the
                      error message. The exception is re-raised afterwards
                      so the caller still sees the original failure.
        """
        assert_system_health(include_llm=True)
        try:
            result_path = self._process_pages(pdf_path=pdf_path)
        except Exception as exc:
            mark_ocr_failed(pdf_path, reason=str(exc))
            raise
        mark_ocr_success(pdf_path)
        return result_path


if __name__ == "__main__":
    COMPANY = "KALYANKJIL"
    YEAR    = "2023"
    DOC_TYPE = "ANNUAL_REPORT"
    filename = f"{COMPANY}/{YEAR}/{DOC_TYPE}/.pdf"
    logger = get_logger("OCRPROCESSOR", filename)

    base_dir   = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, YEAR, DOC_TYPE)
    source_pdf = os.path.join(base_dir, f"{COMPANY}_{YEAR}_{DOC_TYPE}.pdf")

    processor   = OCRProcessor()
    result_path = processor.run(source_pdf, logger)
    print(f"Done → {result_path}")