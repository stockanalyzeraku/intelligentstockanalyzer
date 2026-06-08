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
from logger import get_logger
import pdfplumber


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
        plumber_doc = pdfplumber.open(pdf_path)
    except Exception as exc:
        _pdf_logger.error("Cannot open PDF with pdfplumber", path=pdf_path, error=str(exc))
        plumber_doc = None
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        _pdf_logger.error("Cannot open PDF", path=pdf_path, error=str(exc))
        raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    with doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _extract_with_pdfplumber(plumber_doc, page_num, _pdf_logger)
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
    
def _extract_with_pdfplumber(plumber_doc, page_num: int, logger) -> str:
    """
    Uses pdfplumber to extract text with layout awareness.
    Handles multi-column layouts better than raw fitz.
    """
    if not plumber_doc:
        return ""

    try:
        page = plumber_doc.pages[page_num]

        # ── Extract words with their coordinates ──────────────
        words = page.extract_words(
            x_tolerance     = 3,    # horizontal gap tolerance between words
            y_tolerance     = 3,    # vertical gap tolerance between words
            keep_blank_chars= False,
            use_text_flow   = True  # respects natural reading flow
        )

        if not words:
            return ""

        # ── Detect columns by clustering x positions ──────────
        text = _reconstruct_text_by_columns(words, page.width)
        return text

    except Exception as exc:
        logger.error(f"pdfplumber failed on page {page_num + 1}: {exc}")
        return ""


def _reconstruct_text_by_columns(words: list, page_width: float) -> str:
    """
    Groups words into columns based on x position,
    then reads each column top to bottom.
    """
    if not words:
        return ""

    # ── Sort all words by y position first (top to bottom) ────
    words_sorted = sorted(words, key=lambda w: (round(w["top"] / 10), w["x0"]))

    # ── Detect column boundaries ───────────────────────────────
    # Divide page into left/middle/right thirds
    col1_end = page_width * 0.36   # left column ends at 36% of page
    col2_end = page_width * 0.68   # middle column ends at 68% of page

    col1_words = [w for w in words if w["x0"] < col1_end]
    col2_words = [w for w in words if col1_end <= w["x0"] < col2_end]
    col3_words = [w for w in words if w["x0"] >= col2_end]

    # ── If no clear multi-column layout, return as single flow ─
    if not col2_words and not col3_words:
        return " ".join(w["text"] for w in words_sorted)

    # ── Sort each column top to bottom ────────────────────────
    def words_to_text(col_words):
        if not col_words:
            return ""
        sorted_words = sorted(col_words, key=lambda w: (round(w["top"] / 5), w["x0"]))
        lines        = []
        current_line = []
        current_y    = None

        for word in sorted_words:
            if current_y is None or abs(word["top"] - current_y) < 5:
                current_line.append(word["text"])
                current_y = word["top"]
            else:
                lines.append(" ".join(current_line))
                current_line = [word["text"]]
                current_y    = word["top"]

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    # ── Combine columns with separator ────────────────────────
    col1_text = words_to_text(col1_words)
    col2_text = words_to_text(col2_words)
    col3_text = words_to_text(col3_words)

    sections  = [c for c in [col1_text, col2_text, col3_text] if c.strip()]
    return "\n\n--- COLUMN BREAK ---\n\n".join(sections)

#pc = PageContent(page_number = 1,text = "", char_count  = 0, is_scanned  = False, has_images  = False)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"uploads")
pdf_type, pages = classify_and_extract(os.path.join(path,"KALYANKJIL_ANNUAL_2025.pdf"))
output_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_file")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "KALYANKJIL_ANNUAL_2025_pdfplumber.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for page in pages:
        f.write(f"Page {page.page_number}\n")
        f.write(f"{'-'*50}\n")
        f.write(page.text)
        f.write("\n\n")

print(f"✅ Saved to: {output_file}")
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

