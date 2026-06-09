



# =============================================================================
# CELL 21 — Process Your PDFs
# =============================================================================
"""
Pipeline for processing one or all PDFs in an uploads folder.
PDF naming convention: COMPANYNAME_ANNUAL_REPORT_FY25.pdf
"""

import os
import re
from typing import List, Optional


def _detect_fy(filename: str) -> str:
    """
    Extract fiscal year from a PDF filename.

    Tries to match patterns like FY25, FY2025, 2024-25, 2025.
    Falls back to 'FY00' (unknown) if no pattern found.

    Parameters
    ----------
    filename : str
        Base filename (not full path).

    Returns
    -------
    str
        Normalised FY string (e.g. 'FY25').
    """
    patterns = [
        r"FY(\d{4})",       # FY2025
        r"FY(\d{2})",       # FY25
        r"(\d{4})-\d{2}",   # 2024-25
        r"(\d{4})",         # 2025
    ]
    for pat in patterns:
        m = re.search(pat, filename, re.IGNORECASE)
        if m:
            year_str = m.group(1)
            try:
                return InputValidator.validate_fiscal_year(year_str)
            except ValidationError:
                continue
    return "FY00"


def _detect_scrip(filepath: str) -> str:
    """
    Infer company scrip from the parent folder name.

    The uploads directory structure is: uploads/{Company Name}/file.pdf
    The folder name is treated as the scrip symbol.

    Parameters
    ----------
    filepath : str
        Full path to the PDF file.

    Returns
    -------
    str
        Upper-cased folder name or 'UNKNOWN'.
    """
    parent = os.path.basename(os.path.dirname(filepath))
    if parent and parent != "uploads":
        try:
            return InputValidator.validate_scrip(parent)
        except ValidationError:
            return parent.upper()[:20]
    return "UNKNOWN"


def _clean_text(text: str) -> str:
    """
    Clean raw PDF text: strip page numbers, collapse whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    # Remove lone page numbers on their own line
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def process_single_pdf(pdf_path: str) -> Dict:
    """
    Run the full ingestion pipeline for one PDF file.

    Steps
    -----
    1. Validate path
    2. Idempotency check
    3. Classify + extract text
    4. Extract tables
    5. Clean text
    6. Create chunks
    7. Store chunks (dedup + ChromaDB upsert)
    8. Mark file as processed
    9. Rebuild BM25 index

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    -------
    dict
        {'status', 'scrip', 'fiscal_year', 'chunk_count', 'pdf_type', 'message'}
    """
    _pipe_logger = get_logger("pipeline")

    # Step 1: Validate
    try:
        pdf_path = InputValidator.validate_pdf_path(pdf_path)
    except ValidationError as exc:
        _pipe_logger.error("PDF validation failed", path=pdf_path, error=str(exc))
        return {"status": "error", "message": str(exc)}

    scrip = _detect_scrip(pdf_path)
    fiscal_year = _detect_fy(os.path.basename(pdf_path))

    # Step 2: Idempotency
    if STORAGE_MANAGER.is_file_processed(pdf_path):
        _pipe_logger.info("PDF already processed — skipping.", path=os.path.basename(pdf_path))
        return {"status": "skipped", "scrip": scrip, "fiscal_year": fiscal_year, "message": "Already processed."}

    _pipe_logger.info("Processing PDF.", path=os.path.basename(pdf_path), scrip=scrip, fy=fiscal_year)

    # Step 3: Classify + extract
    try:
        pdf_type, pages = classify_and_extract(pdf_path)
    except ValueError as exc:
        _pipe_logger.error("PDF extraction failed", path=pdf_path, error=str(exc))
        return {"status": "error", "message": str(exc)}

    # Step 4: Extract tables
    tables = extract_tables(pdf_path)

    # Step 5: Clean text
    for page in pages:
        page.text = _clean_text(page.text)

    # Step 6: Create chunks
    chunks = create_chunks(pages, tables, scrip, fiscal_year)
    InputValidator.validate_chunk_count(len(chunks), context=os.path.basename(pdf_path))

    if not chunks:
        _pipe_logger.warning("No chunks created — PDF may be fully scanned.", path=pdf_path)
        return {"status": "warning", "scrip": scrip, "fiscal_year": fiscal_year,
                "chunk_count": 0, "pdf_type": pdf_type.value,
                "message": "No usable text extracted."}

    # Step 7: Store chunks
    stored = STORAGE_MANAGER.store_chunks(chunks)

    # Step 8: Mark processed
    STORAGE_MANAGER.mark_file_processed(pdf_path, scrip, fiscal_year, stored)

    # Step 9: Rebuild BM25
    BM25_INDEX.build()

    _pipe_logger.info(
        "PDF pipeline complete.",
        scrip=scrip, fy=fiscal_year, chunks_stored=stored, pdf_type=pdf_type.value
    )
    return {
        "status": "success",
        "scrip": scrip,
        "fiscal_year": fiscal_year,
        "chunk_count": stored,
        "pdf_type": pdf_type.value,
        "message": f"Stored {stored} chunks.",
    }


def process_company_folder(company_name: str) -> List[Dict]:
    """
    Process all unprocessed PDFs in the uploads/{company_name}/ folder.

    Parameters
    ----------
    company_name : str
        Subfolder name under CONFIG.UPLOADS_PATH.

    Returns
    -------
    List[Dict]
        One result dict per PDF file found.
    """
    _pipe_logger = get_logger("pipeline")
    folder = os.path.join(CONFIG.UPLOADS_PATH, company_name)
    if not os.path.isdir(folder):
        _pipe_logger.error("Company folder not found", folder=folder)
        return [{"status": "error", "message": f"Folder not found: {folder}"}]

    pdf_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        _pipe_logger.warning("No PDF files found in folder.", folder=folder)
        return [{"status": "warning", "message": "No PDFs found."}]

    _pipe_logger.info("Processing company folder.", company=company_name, pdf_count=len(pdf_files))
    results = []
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path)
        results.append(result)
        print(f"  [{result['status'].upper()}] {os.path.basename(pdf_path)} — {result.get('message', '')}")

    return results


# ── Quick-start helper printed to Colab output ─────────────────────────────
print("""
[Cell 21] Pipeline ready.

Usage:
  # Process a single PDF:
  result = process_single_pdf('/content/drive/MyDrive/brain/uploads/RELIANCE/ANNUAL_REPORT_FY25.pdf')

  # Process all PDFs for a company:
  results = process_company_folder('RELIANCE')

  # Then query:
  answer = AGENT.ask('What was the revenue growth in FY25?', scrip='RELIANCE')
  print(answer['answer'])
""")

# ----------------------------------------------------------------------------
# Cell 21: Process Your PDFs
# Purpose: Full ingestion pipeline — validate, extract, chunk, store, index.
# Key Classes: None
# Key Functions:
#   process_single_pdf(pdf_path) → dict
#   process_company_folder(company_name) → List[Dict]
#   _detect_fy(filename) → str
#   _detect_scrip(filepath) → str
#   _clean_text(text) → str
# Key Constants/Config: CONFIG.UPLOADS_PATH
# Imports exported: process_single_pdf, process_company_folder
# Depends on: Cells 3–14 (all pipeline components), STORAGE_MANAGER, BM25_INDEX
# Critical notes: PDF naming convention expected: {SCRIP}_ANNUAL_REPORT_FY25.pdf
#   Folder name under uploads/ is used as the scrip symbol.
#   BM25 is rebuilt after EVERY successful process_single_pdf() call.
#   process_single_pdf() is idempotent — safe to re-run.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

