# tests/conftest.py
"""
Shared pytest fixtures for the ocrprocessor test suite.

The key challenge: ALLOWED_BASE is a module-level constant computed at
import time from CONFIG.UPLOADS_PATH. Every test that calls _validate_filepath
or _validate_output_path must patch ALLOWED_BASE to point at a controlled
temporary directory, or tests will try to write into the real uploads folder.

Pattern used throughout: the `allowed_base` fixture patches both
skelton.ALLOWED_BASE and validator.ALLOWED_BASE simultaneously.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Filesystem fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def allowed_base(tmp_path, monkeypatch):
    """
    Override ALLOWED_BASE with a temp directory.
    Must be used by every test that exercises filepath validation.
    """
    base = tmp_path / "uploads"
    base.mkdir()
    monkeypatch.setattr("codebase.ocrprocessor.skelton.ALLOWED_BASE", base)
    monkeypatch.setattr("codebase.ocrprocessor.validator.ALLOWED_BASE", base)
    return base


@pytest.fixture
def valid_pdf(allowed_base):
    """
    Minimal valid PDF file at the correct directory depth:
    uploads/<company>/<year>/<company>_<type>_<year>.pdf
    """
    company_dir = allowed_base / "TESTCO" / "2024"
    company_dir.mkdir(parents=True)
    pdf_path = company_dir / "TESTCO_ANNUAL_2024.pdf"
    # Real PDF header — validator checks for b"%PDF-"
    pdf_path.write_bytes(b"%PDF-1.4 minimal test fixture\n%%EOF")
    return pdf_path


@pytest.fixture
def valid_json(allowed_base):
    """Minimal valid JSON file at the correct directory depth."""
    company_dir = allowed_base / "TESTCO" / "2024"
    company_dir.mkdir(parents=True, exist_ok=True)
    json_path = company_dir / "TESTCO_ANNUAL_2024.json"
    json_path.write_text(
        json.dumps([{"page_number": 1, "text": "Annual report content."}]),
        encoding="utf-8",
    )
    return json_path


@pytest.fixture
def outside_path(tmp_path):
    """A real file that exists but is outside ALLOWED_BASE — path traversal target."""
    evil = tmp_path / "outside" / "secret.pdf"
    evil.parent.mkdir(parents=True)
    evil.write_bytes(b"%PDF-1.4 should never be accessible\n%%EOF")
    return evil


# ── OCR text fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def clean_financial_text():
    """
    Representative OCR output from a real annual report page.
    Must pass all validation checks — used to confirm no false positives.
    """
    return (
        "ANNUAL REPORT 2023-24\n\n"
        "The Board of Directors presents the Annual Report and Audited\n"
        "Financial Statements for the year ended 31st March 2024.\n\n"
        "Revenue from Operations: ₹8,453.21 Crores\n"
        "Profit After Tax: ₹621.18 Crores\n"
        "EBITDA Margin: 11.2% (FY23: 9.8%)\n"
        "Earnings Per Share: ₹62.11\n\n"
        "The company has maintained a consistent dividend payout policy.\n"
        "Your Directors recommend a dividend of ₹4.00 per equity share.\n"
    )


# ── External dependency mocks ────────────────────────────────────────────────

@pytest.fixture
def mock_mistral_response():
    """Fake OCR response from Mistral API."""
    page = MagicMock()
    page.markdown = "Page content from OCR."
    response = MagicMock()
    response.pages = [page]
    return response


@pytest.fixture
def mock_mistral_client(mock_mistral_response):
    """Mistral client that returns a successful OCR response."""
    client = MagicMock()
    client.ocr.process.return_value = mock_mistral_response
    return client


@pytest.fixture
def mock_fitz_doc():
    """PyMuPDF document with two pages."""
    page = MagicMock()
    pix = MagicMock()
    pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    page.get_pixmap.return_value = pix

    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=2)
    doc.__getitem__ = MagicMock(return_value=page)
    return doc