from __future__ import annotations

import json
import os
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from mistralai.client import Mistral
import base64

from config import CONFIG
from logger import get_logger
from healthcheck import assert_system_health
from codebase.mistralaiprocessor.skelton import PageContent
from codebase.mistralaiprocessor.validator import _validate_filepath, _validate_ocr_text
logger = get_logger(__name__)

class MistralAIProcessor:
    """Orchestrates the full PDF → OCR → JSON pipeline using Mistral."""

    def __init__(self, pdf_path: str, output_file: str, api_key: Optional[str] = None) -> None:

        # Use provided key or fall back to config — never accept a hardcoded literal
        self._api_key = api_key or CONFIG.MISTRAL_API_KEY
        self._pages: list[PageContent] = []
        self._client: Optional[Mistral] = None

    def _init_client(self) -> None:
        self.log.info("Creating Mistral client")
        self._client = Mistral(api_key=self._api_key)
        self.log.info("Mistral client ready")

    def _process_pages(self, pdf_path: str) -> None:
        _validate_filepath(pdf_path)
        pdf_name = Path(pdf_path).name
        self.log.info(f"Opening PDF: '{pdf_name}'")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_idx in range(total_pages):
            pix = doc[page_idx].get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            ocr_response = self._client.ocr.process(
                model=CONFIG.MISTRAL_MODEL_OCR,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}",
                },
            )
            page_markdown = ocr_response.pages[0].markdown if ocr_response.pages else ""
            self._pages.append(PageContent(page_number=page_idx + 1, text=_validate_ocr_text(page_markdown)))
            self.log.info(f"Page {page_idx + 1}/{total_pages} OCR complete — {len(page_markdown)} chars")

        doc.close()
        self.save_pages_to_json(self._pages, self._output_file)

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> None:
            """Serialise a list of PageContent objects to a JSON file."""
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data = [asdict(pc) for pc in pages]
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            self.log.info(f"Pages saved to JSON → {output_path} ({len(pages)} pages)")
    
    def run(self, pdf_path:str) -> str:
        assert_system_health(include_llm=True)
        with self.log.timed("ocr_pipeline", pdf_path=pdf_path, output_file=self._output_file):
            self._init_client()
            self._process_pages()
        self.log.process_event("ocr_pipeline_completed", "ocr", output_file=self._output_file)
        return self._output_file


if __name__ == "__main__":
    COMPANY = "KALYANKJIL"
    YEAR = 2023
    DOC_TYPE = "ANNUAL"

    base_dir = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}")
    source_pdf = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.pdf")
    output_json = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.json")

    processor = MistralAIProcessor()
    result_path = processor.run(source_pdf)
    print(f"Done → {result_path}")
