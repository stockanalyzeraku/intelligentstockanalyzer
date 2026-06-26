from __future__ import annotations

import json
import os

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from mistralai.client import Mistral
import base64

from config import CONFIG
from logger import get_logger
from healthcheck import assert_system_health
from codebase.ocrprocessor.skelton import PageContent
from codebase.ocrprocessor.validator import (
    _validate_ocr_text,
    _validate_filepath,
    _validate_output_path

)
logger = get_logger(__name__)

class OCRProcessor:
    """Orchestrates the full PDF → OCR → JSON pipeline using Mistral."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or CONFIG.MISTRAL_API_KEY
        self._pages: list[PageContent] = []
        self._client: Optional[Mistral] = None

    def _init_client(self) -> None:
        self._client = Mistral(
            api_key=self._api_key,
            timeout = 30
            )
        del self._api_key

    def _process_pages(self, pdf_path: str) -> Path:
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
        return self.save_pages_to_json(self._pages, os.path.splitext(Path(pdf_path))[0] + ".json")

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> None:
        resolved_output_path = _validate_output_path(output_path)
        parent = Path(output_path).parent
        if str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        data = [asdict((pc)) for pc in pages]
        with open(resolved_output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return output_path
    
    def run(self, pdf_path:str) -> str:
        assert_system_health(include_llm=True)
        self._init_client()
        output_path = self._process_pages(pdf_path=pdf_path)
        return output_path


if __name__ == "__main__":
    COMPANY = "KALYANKJIL"
    YEAR = 2023
    DOC_TYPE = "ANNUAL"

    base_dir = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}")
    source_pdf = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.pdf")
    output_json = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.json")

    processor = OCRProcessor()
    result_path = processor.run(source_pdf)
    print(f"Done → {result_path}")
