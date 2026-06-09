# =============================================================================
# CELL 11 — Table Extractor
# =============================================================================
"""
Extract tables from PDF files using pdfplumber.
Outputs structured text in "Header: Value | Header: Value" format.
Financial tables are flagged for routing to the financial_facts collection.
"""

from typing import Dict, List, Optional
import re
from logger import get_logger
import os


def _has_numeric_density(cells: List[str], threshold: float = 0.5) -> bool:
    """
    Return True if a majority of non-empty cells contain numbers.

    Parameters
    ----------
    cells : List[str]
        Cell texts from one row.
    threshold : float
        Fraction of numeric cells required.
    """
    non_empty = [c for c in cells if c and c.strip()]
    if not non_empty:
        return False
    numeric = [c for c in non_empty if re.search(r"\d", c)]
    return len(numeric) / len(non_empty) >= threshold


def _reconstruct(table_data: List[List[Optional[str]]], page_number: int) -> Optional[Dict]:
    """
    Convert a 2-D cell list (from pdfplumber) into structured text.

    Parameters
    ----------
    table_data : List[List[Optional[str]]]
        Raw table cells from pdfplumber.
    page_number : int
        Source page (1-based).

    Returns
    -------
    dict or None
        {'text', 'page_number', 'is_financial'} or None on invalid table.
    """
    if not table_data or len(table_data) < 2:
        return None

    # Normalise cells
    cleaned: List[List[str]] = [
        [str(c).strip() if c else "" for c in row]
        for row in table_data
    ]

    # First row as header; if it has numeric density, treat row 0 as data too
    header_row = cleaned[0]
    data_rows = cleaned[1:]
    is_financial = _has_numeric_density(
        [cell for row in data_rows[:5] for cell in row]
    )

    lines: List[str] = []
    for row in data_rows:
        if not any(cell for cell in row):
            continue  # skip empty rows
        pairs = []
        for h, v in zip(header_row, row):
            if h or v:
                pairs.append(f"{h}: {v}" if h else v)
        if pairs:
            lines.append(" | ".join(pairs))

    if not lines:
        return None

    return {
        "text": "\n".join(lines),
        "page_number": page_number,
        "is_financial": is_financial,
    }


def extract_tables(pdf_path: str) -> List[Dict]:
    """
    Extract all tables from a PDF file.

    Parameters
    ----------
    pdf_path : str
        Validated absolute path to the PDF.

    Returns
    -------
    List[Dict]
        Each dict: {'text', 'page_number', 'is_financial'}.
        Empty list if no tables found or pdfplumber unavailable.
    """
    _tbl_logger = get_logger("table_extractor")
    import pdfplumber

    results: List[Dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    for table_data in tables:
                        reconstructed = _reconstruct(table_data, page.page_number)
                        if reconstructed:
                            results.append(reconstructed)
                except Exception as exc:
                    _tbl_logger.warning(
                        "Failed to extract tables from page",
                        page=page.page_number,
                        error=str(exc),
                    )
    except Exception as exc:
        _tbl_logger.error("pdfplumber failed", path=pdf_path, error=str(exc))
        return []

    _tbl_logger.info(
        "Table extraction complete.",
        path=os.path.basename(pdf_path),
        table_count=len(results),
    )
    return results

# ----------------------------------------------------------------------------
# Cell 11: Table Extractor
# Purpose: Extract and restructure tables from PDFs as "Header: Value" text.
# Key Classes: None
# Key Functions:
#   extract_tables(pdf_path) → List[Dict]
#   _reconstruct(table_data, page_number) → dict | None
#   _has_numeric_density(cells, threshold) → bool
# Key Constants/Config: numeric density threshold=0.5 (inline)
# Imports exported: extract_tables
# Depends on: Cell 4 (get_logger)
# Critical notes: is_financial flag drives routing to COL_FACTS in chunker.
#   _reconstruct returns None for tables with <2 rows or no content.
#   pdfplumber may miss some tables in complex layouts — treat as best-effort.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
