"""
End-to-end Mistral OCR ingestion pipeline for annual report PDFs.

Responsibilities:
1. Accept a source PDF path and output paths.
2. Render each PDF page to a PNG image (via PyMuPDF / fitz) at 200 DPI.
3. Upload each image to Mistral and run OCR to obtain markdown per page.
4. Collect all pages into a list of PageContent objects.
5. Persist the raw OCR output to JSON for downstream processing.
"""

from __future__ import annotations

import json
import os
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from mistralai.client import Mistral

from config import CONFIG
from inputvalidator import InputValidator
from logger import get_logger
from healthcheck import assert_system_health

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Raw OCR content extracted from one page of the source PDF."""
    page_num: int
    text: str


class MistralAIProcessor:
    """Orchestrates the full PDF → OCR → JSON pipeline using Mistral."""

    def __init__(
        self,
        pdf_path: str,
        output_file: str,
        api_key: Optional[str] = None,
    ) -> None:
        self.log = get_logger(self.__class__.__name__)
        self.log.info("Initialising MistralAIProcessor")

        # Use provided key or fall back to config — never accept a hardcoded literal
        self._api_key = api_key or CONFIG.MISTRAL_API_KEY

        validator = InputValidator()
        self._pdf_path = validator.validate_pdf_path(pdf_path)
        self._output_file = validator.validate_output_path(output_file)

        self._pages: list[PageContent] = []
        self._client: Optional[Mistral] = None

        self.log.info(f"Config — source PDF : '{self._pdf_path}'")
        self.log.info(f"Config — output JSON: '{self._output_file}'")

    def run(self) -> str:
        """Execute the full OCR pipeline. Returns the output JSON path."""
        assert_system_health(include_llm=True)
        self.log.process_event(
            "ocr_pipeline_started", "ocr",
            pdf_path=self._pdf_path, output_file=self._output_file,
        )
        with self.log.timed("ocr_pipeline", pdf_path=self._pdf_path, output_file=self._output_file):
            self._init_client()
            self._process_pages()
        self.log.process_event("ocr_pipeline_completed", "ocr", output_file=self._output_file)
        return self._output_file

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> None:
        """Serialise a list of PageContent objects to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [asdict(pc) for pc in pages]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        self.log.info(f"Pages saved to JSON → {output_path} ({len(pages)} pages)")

    def _init_client(self) -> None:
        self.log.info("Creating Mistral client")
        self._client = Mistral(api_key=self._api_key)
        self.log.info("Mistral client ready")

    def _process_pages(self) -> None:
        """Render every PDF page to PNG (base64) and run OCR via Mistral."""
        import base64

        pdf_name = Path(self._pdf_path).name
        self.log.info(f"Opening PDF: '{pdf_name}'")

        doc = fitz.open(self._pdf_path)
        total_pages = len(doc)
        self.log.info(f"PDF opened — {total_pages} pages to process")

        for page_idx in range(total_pages):
            pix = doc[page_idx].get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            ocr_response = self._client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}",
                },
            )
            page_markdown = ocr_response.pages[0].markdown if ocr_response.pages else ""
            self._pages.append(PageContent(page_num=page_idx + 1, text=page_markdown))
            self.log.info(f"Page {page_idx + 1}/{total_pages} OCR complete — {len(page_markdown)} chars")

        doc.close()
        self.save_pages_to_json(self._pages, self._output_file)


if __name__ == "__main__":
    COMPANY = "KALYANKJIL"
    YEAR = 2023
    DOC_TYPE = "ANNUAL"

    base_dir = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}")
    source_pdf = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.pdf")
    output_json = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.json")

    processor = MistralAIProcessor(
        pdf_path=source_pdf,
        output_file=output_json,
        # api_key is read from CONFIG.MISTRAL_API_KEY / .env automatically
    )
    result_path = processor.run()
    print(f"Done → {result_path}")
