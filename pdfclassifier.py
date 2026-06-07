# =============================================================================
# CELL 9 — PDF Classifier
# =============================================================================
"""
Classify PDFs as NATIVE, SCANNED, or MIXED and extract per-page content.
Scanned pages are identified by sparse text + embedded images.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class PDFType(Enum):
    """Enumeration of PDF content types."""
    NATIVE = "native"    # Selectable text throughout
    SCANNED = "scanned"  # Image-only pages (OCR needed)
    MIXED = "mixed"      # Combination of native and scanned pages


@dataclass
class PageContent:
    """
    Content extracted from one PDF page.

    Attributes
    ----------
    page_number : int
        1-based page index.
    text : str
        Extracted text (empty for scanned pages).
    char_count : int
        Number of characters in text.
    is_scanned : bool
        True if page is image-only.
    has_images : bool
        True if page contains embedded images.
    """
    page_number: int
    text: str
    char_count: int
    is_scanned: bool
    has_images: bool

    @property
    def is_usable(self) -> bool:
        """Return True if the page contains usable text content."""
        return not self.is_scanned and self.char_count > 50


def _is_page_scanned(text: str, has_images: bool, min_chars: int = 100) -> bool:
    """
    Determine if a PDF page is scanned (image-based).

    A page is considered scanned when it has fewer characters than
    min_chars AND contains embedded images.

    Parameters
    ----------
    text : str
        Raw extracted text.
    has_images : bool
        Whether the page contains embedded image objects.
    min_chars : int
        Minimum character count to consider a page as native text.

    Returns
    -------
    bool
    """
    return len(text.strip()) < min_chars and has_images


def classify_and_extract(pdf_path: str) -> Tuple[PDFType, List[PageContent]]:
    """
    Open a PDF, classify it, and extract per-page content.

    Parameters
    ----------
    pdf_path : str
        Validated absolute path to the PDF file.

    Returns
    -------
    Tuple[PDFType, List[PageContent]]
        Overall PDF type and list of PageContent objects.

    Raises
    ------
    ValueError
        If the PDF cannot be opened.
    """
    _pdf_logger = get_logger("pdf_classifier")
    import fitz  # PyMuPDF

    pages: List[PageContent] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        _pdf_logger.error("Cannot open PDF", path=pdf_path, error=str(exc))
        raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    with doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text") or ""
            image_list = page.get_images(full=False)
            has_images = len(image_list) > 0
            is_scanned = _is_page_scanned(text, has_images)
            pages.append(
                PageContent(
                    page_number=page_num + 1,
                    text=text,
                    char_count=len(text.strip()),
                    is_scanned=is_scanned,
                    has_images=has_images,
                )
            )

    scanned_count = sum(1 for p in pages if p.is_scanned)
    native_count = len(pages) - scanned_count

    if scanned_count == 0:
        pdf_type = PDFType.NATIVE
    elif native_count == 0:
        pdf_type = PDFType.SCANNED
    else:
        pdf_type = PDFType.MIXED

    _pdf_logger.info(
        "PDF classified.",
        path=os.path.basename(pdf_path),
        type=pdf_type.value,
        total_pages=len(pages),
        scanned_pages=scanned_count,
        native_pages=native_count,
    )
    return pdf_type, pages

# ----------------------------------------------------------------------------
# Cell 9: PDF Classifier
# Purpose: Classify PDFs as NATIVE/SCANNED/MIXED and extract page content.
# Key Classes: PDFType (enum), PageContent (dataclass)
# Key Functions:
#   classify_and_extract(pdf_path) → Tuple[PDFType, List[PageContent]]
#   _is_page_scanned(text, has_images, min_chars) → bool
# Key Constants/Config: min_chars=100 (inline default, change via Config if needed)
# Imports exported: PDFType, PageContent, classify_and_extract
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: Uses PyMuPDF (fitz). PageContent.is_usable filters out
#   scanned pages and near-empty pages before chunking.
#   SCANNED PDFs will still be processed but produce few usable chunks.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

