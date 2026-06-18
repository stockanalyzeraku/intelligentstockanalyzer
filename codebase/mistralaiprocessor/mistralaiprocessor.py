"""
mistralaiprocessor.py
=====================
End-to-end Mistral OCR ingestion pipeline for annual report PDFs.

Responsibilities
----------------
1. Accept a source PDF path and output paths.
2. Render each PDF page to a PNG image (via PyMuPDF / fitz) at 200 DPI.
3. Upload each image to Mistral and run OCR to obtain markdown per page.
4. Collect all pages into a list of :class:`PageContent` objects.
5. Persist the raw OCR output to JSON for downstream processing.

This module stays in the project root because it is an *ingestion* module,
not a *cleaning* module.  The cleaning package (``cleaning/``) is invoked
separately on the JSON output produced here.

Folder convention
-----------------
All output artefacts follow the project path convention::

    uploads/<CompanyName>/<DocType>_<Year>/<FileName>

For example::

    uploads/KALYANKJIL/ANNUAL_2025/KALYANKJIL_ANNUAL_2025.json
"""

from __future__ import annotations

import json
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from mistralai.client import Mistral

from config import CONFIG
from inputvalidator import InputValidator
from logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PageContent
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """
    Raw OCR content extracted from one page of the source PDF.

    Attributes
    ----------
    page_num : int
        1-based page index.
    text : str
        Markdown text returned by Mistral OCR for this page.
    """

    page_num: int
    text:     str


# ---------------------------------------------------------------------------
# MistralAIProcessor
# ---------------------------------------------------------------------------

class MistralAIProcessor:
    """
    Orchestrates the full PDF → OCR → JSON pipeline using Mistral.

    Parameters
    ----------
    api_key : str | None
        Mistral API key.  Falls back to ``CONFIG.MISTRAL_API_KEY`` when
        ``None`` or an empty string.
    pdf_path : str
        Absolute path to the source PDF file.
    output_file : str
        Absolute path for the JSON artefact produced by :meth:`run`.
    """

    def __init__(self, api_key:     Optional[str], pdf_path:str, output_file: str) -> None:
    
        self.log = get_logger(self.__class__.__name__)
        self.log.info("Initialising MistralAIProcessor")
        self._api_key = api_key

        validator         = InputValidator()
        self._pdf_path    = validator.validate_pdf_path(pdf_path)
        self._output_file = validator.validate_output_path(output_file)

        self._pages:          list[PageContent] = []
        self._client:         Optional[Mistral] = None
        self._markdown_text:  str               = ""

        self.log.info(
            f"Config — source PDF : '{self._pdf_path}'"
        )
        self.log.info(
            f"Config — output JSON: '{self._output_file}'"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        Execute the full OCR pipeline.

        Steps
        -----
        1. Initialise the Mistral client.
        2. Render each PDF page to PNG and upload to Mistral.
        3. Run OCR on each page image.
        4. Save all pages to JSON.

        Returns
        -------
        str
            Path to the generated JSON output file.
        """
        self.log.info("Pipeline started", event="ocr_pipeline_started")
        with self.log.timed("ocr_pipeline", pdf_path=self._pdf_path, output_file=self._output_file):
            self._init_client()
            self._upload_pdf_and_process_ocr()
        self.log.info(f"Pipeline complete — output: '{self._output_file}'", event="ocr_pipeline_completed")
        return self._output_file

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> None:
        """
        Serialise a list of :class:`PageContent` objects to a JSON file.

        Parameters
        ----------
        pages       : list[PageContent]
        output_path : str — destination file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [asdict(pc) for pc in pages]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        self.log.info(f"Pages saved to JSON → {output_path} ({len(pages)} pages)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Create and store the authenticated Mistral client."""
        self.log.info("Creating Mistral client")
        self._client = Mistral(api_key=self._api_key)
        self.log.info("Mistral client ready")

    # def _get_signed_url(self, file_id: str) -> str:
    #     """
    #     Retrieve a temporary signed URL for an already-uploaded file.

    #     Parameters
    #     ----------
    #     file_id : str
    #         The Mistral file ID returned after upload.

    #     Returns
    #     -------
    #     str
    #         Signed download URL valid for a short window.
    #     """
    #     self.log.info(f"Fetching signed URL — file_id: {file_id}")
    #     signed = self._client.files.get_signed_url(file_id=file_id)
    #     self.log.info("Signed URL obtained")
    #     return signed.url

    # def _upload_pdf_and_process_ocr(self) -> None:
    #     """
    #     Render every PDF page to a PNG, upload to Mistral, and run OCR.

    #     Each page is processed independently so that a failure on one page
    #     does not abort the entire document.  The last successfully uploaded
    #     ``file_id`` is used for logging purposes only.

    #     Raises
    #     ------
    #     RuntimeError
    #         If the source PDF cannot be opened by PyMuPDF.
    #     """
    #     pdf_name = Path(self._pdf_path).name
    #     self.log.info(f"Opening PDF: '{pdf_name}'")

    #     doc = fitz.open(self._pdf_path)
    #     total_pages = len(doc)
    #     self.log.info(f"PDF opened — {total_pages} pages to process")

    #     last_file_id: Optional[str] = None

    #     for page_idx in range(total_pages):
    #         page_num_1based = page_idx + 1
    #         self.log.info(f"Processing page {page_num_1based}/{total_pages}")

    #         # Render page to PNG bytes at 200 DPI
    #         page    = doc[page_idx]
    #         pix     = page.get_pixmap(dpi=200)
    #         img_bytes = pix.tobytes("png")

    #         # Upload rendered image to Mistral
    #         uploaded = self._client.files.upload(
    #             file={"file_name": pdf_name, "content": img_bytes},
    #             purpose="ocr",
    #         )
    #         last_file_id = uploaded.id
    #         self.log.info(
    #             f"Page {page_num_1based} uploaded — file_id: {uploaded.id}"
    #         )

    #         sign_url = self._get_signed_url(uploaded.id)
    #         self._process_ocr(sign_url, page_num_1based)

    #     doc.close()
    #     self.log.info(
    #         f"All {total_pages} pages processed — last file_id: {last_file_id}"
    #     )

    #     self.save_pages_to_json(self._pages, self._output_file)


    def _upload_pdf_and_process_ocr(self):
        import base64

        pdf_name = Path(self._pdf_path).name
        self.log.info(f"Opening PDF: '{pdf_name}'")

        doc = fitz.open(self._pdf_path)
        total_pages = len(doc)
        self.log.info(f"PDF opened — {total_pages} pages to process")

        last_file_id: Optional[str] = None

        for page_idx in range(total_pages):
        
        # ... your existing page loop ...
            pix = doc[page_idx].get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
    
        # Skip upload — send base64 directly to OCR
            b64 = base64.b64encode(img_bytes).decode("utf-8")
    
            ocr_response = self._client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}"
                }
            )
            page_markdown = ocr_response.pages[0].markdown if ocr_response.pages else ""
            self._pages.append(PageContent(page_num=page_idx+1, text=page_markdown))
            print(f"Page {page_idx+1} OCR complete — {len(page_markdown)} characters")
        doc.close()
        self.save_pages_to_json(self._pages, self._output_file)


    # def _process_ocr(self, image_url: str, page_num: int) -> None:
    #     """
    #     Call the Mistral OCR endpoint for one page and append the result.

    #     Parameters
    #     ----------
    #     image_url : str
    #         Signed URL of the uploaded page image.
    #     page_num : int
    #         1-based page number (stored in the resulting :class:`PageContent`).
    #     """
    #     self.log.info(
    #         f"Running OCR — page {page_num}, model '{CONFIG.MISTRAL_MODEL_OCR}'"
    #     )

    #     ocr_response = self._client.ocr.process(
    #         model    = "mistral-ocr-latest",
    #         document = {"type": "image_url", "image_url": image_url},
    #         )
    #     page_markdown = ocr_response.pages[0].markdown if ocr_response.pages else ""
    #     self._pages.append(PageContent(page_num=page_num, text=page_markdown))

    #     # Keep a running concatenation of all markdown (useful for debug)
    #     self._markdown_text += f"\n\n---\n\n{page_markdown}"

    #     self.log.info(
    #         f"OCR complete — page {page_num}, "
    #         f"{len(page_markdown):,} chars returned"
    #     )


# ---------------------------------------------------------------------------
# Entry point — smoke test / CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    COMPANY  = "KALYANKJIL"
    YEAR     = 2023
    DOC_TYPE = "ANNUAL"

    base_dir    = os.path.join(CONFIG.UPLOADS_PATH, COMPANY, f"{DOC_TYPE}_{YEAR}")
    source_pdf  = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.pdf")
    output_json = os.path.join(base_dir, f"{COMPANY}_{DOC_TYPE}_{YEAR}.json")

    processor = MistralAIProcessor(
        api_key="JUtavgCRQvS9p96wPLi9Nw8z2IIHgYCS",          # reads from CONFIG.MISTRAL_API_KEY / .env
        pdf_path=source_pdf,
        output_file=output_json,
    )
    result_path = processor.run()
    print(f"Done → {result_path}")