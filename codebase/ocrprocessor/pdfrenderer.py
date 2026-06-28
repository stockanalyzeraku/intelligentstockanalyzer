"""
pdf_renderer.py — PDF page rendering using PyMuPDF (fitz).

Owns two things only:
  1. Opening a PDF and iterating its pages.
  2. Rendering each page to raw PNG bytes.

Nothing about Mistral, OCR text, or file writing lives here.
"""
from __future__ import annotations

from typing import Iterator

import fitz  # PyMuPDF

from codebase.ocrprocessor.skelton import MAX_PDF_PAGES


class PDFRenderer:
    """
    Renders each page of a PDF to a PNG image using PyMuPDF.

    Enforces MAX_PDF_PAGES to prevent memory exhaustion on large files.
    The document is always closed via try/finally, even on error mid-loop.

    Args:
        dpi: Render resolution. 200 DPI gives good OCR accuracy
             without excessive memory use.
    """

    def __init__(self, dpi: int = 200) -> None:
        self._dpi = dpi

    def render_pages(self, pdf_path: str) -> Iterator[tuple[int, bytes]]:
        """
        Yield (page_number, png_bytes) for every page in the PDF.

        Args:
            pdf_path: Absolute path to the source PDF file.

        Yields:
            Tuple of (1-indexed page number, raw PNG bytes).

        Raises:
            ValueError: If the PDF exceeds MAX_PDF_PAGES.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if total_pages > MAX_PDF_PAGES:
            doc.close()
            raise ValueError(
                f"PDF has {total_pages} pages; "
                f"maximum allowed is {MAX_PDF_PAGES}. "
                "Split the document before processing."
            )

        try:
            for page_idx in range(total_pages):
                pix = doc[page_idx].get_pixmap(dpi=self._dpi)
                yield page_idx + 1, pix.tobytes("png")
        finally:
            doc.close()