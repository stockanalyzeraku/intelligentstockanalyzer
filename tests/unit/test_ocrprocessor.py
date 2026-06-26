# tests/unit/test_ocrprocessor.py
"""
Tests for OCRProcessor.

ALL external dependencies are mocked:
  - Mistral client       → unittest.mock.MagicMock
  - fitz (PyMuPDF)       → unittest.mock.patch
  - filesystem           → tmp_path + allowed_base fixture
  - assert_system_health → patched to no-op
  - get_logger           → patched to MagicMock

This ensures tests are fast (no PDF rendering), deterministic (no API calls),
and test only the OCRProcessor logic itself.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from codebase.ocrprocessor.ocrprocessor import OCRProcessor
from codebase.ocrprocessor.skelton import PageContent


# ── Fixtures specific to OCRProcessor ───────────────────────────────────────

@pytest.fixture
def processor(monkeypatch):
    """
    OCRProcessor with all external dependencies patched.
    Uses a fake API key so _init_client doesn't need real credentials.
    """
    monkeypatch.setattr("codebase.ocrprocessor.ocrprocessor.assert_system_health",
                        lambda **kwargs: None)
    with patch("codebase.ocrprocessor.ocrprocessor.get_logger") as mock_logger:
        mock_logger.return_value = MagicMock()
        proc = OCRProcessor(api_key="test-api-key-fake")
    return proc


@pytest.fixture
def processor_with_client(processor, mock_mistral_client, monkeypatch):
    """OCRProcessor with a pre-built mock Mistral client."""
    with patch("codebase.ocrprocessor.ocrprocessor.Mistral",
               return_value=mock_mistral_client):
        processor._init_client()
    return processor


# ── __init__ ────────────────────────────────────────────────────────────────

class TestOCRProcessorInit:

    def test_logger_is_set_on_init(self, processor):
        # C-01 fix: self.log must be set
        assert hasattr(processor, "log"), \
            "C-01 NOT FIXED: self.log is never assigned in __init__"

    def test_client_is_none_before_init(self, processor):
        assert processor._client is None

    def test_api_key_is_stored(self, processor):
        assert processor._api_key == "test-api-key-fake"

    def test_custom_api_key_used_over_config(self):
        with patch("codebase.ocrprocessor.ocrprocessor.get_logger", return_value=MagicMock()):
            proc = OCRProcessor(api_key="custom-key")
        assert proc._api_key == "custom-key"


# ── _init_client ─────────────────────────────────────────────────────────────

class TestInitClient:

    def test_client_is_set_after_init(self, processor, mock_mistral_client):
        with patch("codebase.ocrprocessor.ocrprocessor.Mistral",
                   return_value=mock_mistral_client):
            processor._init_client()
        assert processor._client is mock_mistral_client

    def test_api_key_deleted_after_client_built(self, processor, mock_mistral_client):
        """C-03 fix: del self._api_key (not self._api_key_ref)."""
        with patch("codebase.ocrprocessor.ocrprocessor.Mistral",
                   return_value=mock_mistral_client):
            processor._init_client()
        assert not hasattr(processor, "_api_key"), \
            "C-03 NOT FIXED: API key should be deleted after client is built"

    def test_mistral_configured_with_timeout(self, processor):
        """Timeout must be set — H-11 fix."""
        with patch("codebase.ocrprocessor.ocrprocessor.Mistral") as mock_cls:
            mock_cls.return_value = MagicMock()
            processor._init_client()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("timeout") == 30, \
            "Mistral client must be configured with timeout=30"


# ── save_pages_to_json ────────────────────────────────────────────────────────

class TestSavePagesToJson:

    def test_writes_valid_json_file(self, processor, allowed_base):
        output_path = allowed_base / "TESTCO" / "2024" / "output.json"
        output_path.parent.mkdir(parents=True)
        pages = [
            PageContent(page_number=1, text="Page one content."),
            PageContent(page_number=2, text="Page two content."),
        ]
        processor.save_pages_to_json(pages, str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["page_number"] == 1
        assert data[0]["text"] == "Page one content."

    def test_creates_parent_directory(self, processor, allowed_base):
        deep_path = allowed_base / "TESTCO" / "2024" / "subdir" / "out.json"
        pages = [PageContent(page_number=1, text="Content.")]
        processor.save_pages_to_json(pages, str(deep_path))
        assert deep_path.exists()

    def test_output_outside_allowed_base_raises(self, processor, tmp_path):
        """C-04 fix: output path must be validated against ALLOWED_BASE."""
        evil_path = tmp_path / "outside" / "output.json"
        evil_path.parent.mkdir()
        pages = [PageContent(page_number=1, text="Content.")]
        with pytest.raises(Exception):  # FilePathError or ValueError
            processor.save_pages_to_json(pages, str(evil_path))

    def test_pages_serialized_as_dataclass_fields(self, processor, allowed_base):
        """C-05 fix: asdict(pc) not asdict(_validate_ocr_text(pc))."""
        output = allowed_base / "TESTCO" / "2024" / "out.json"
        output.parent.mkdir(parents=True)
        pages = [PageContent(page_number=1, text="Hello.")]
        processor.save_pages_to_json(pages, str(output))
        data = json.loads(output.read_text())
        assert "page_number" in data[0]
        assert "text" in data[0]


# ── _process_pages ────────────────────────────────────────────────────────────

class TestProcessPages:

    def test_pages_reset_each_call(
        self, processor_with_client, valid_pdf, mock_fitz_doc, allowed_base
    ):
        """C-07 fix: pages must not accumulate across runs."""
        output1 = allowed_base / "TESTCO" / "2024" / "out1.json"
        output2 = allowed_base / "TESTCO" / "2024" / "out2.json"
        output1.parent.mkdir(parents=True, exist_ok=True)

        with patch("codebase.ocrprocessor.ocrprocessor.fitz.open",
                   return_value=mock_fitz_doc):
            processor_with_client._process_pages(str(valid_pdf), str(output1))
            pages_after_first = json.loads(output1.read_text())
            processor_with_client._process_pages(str(valid_pdf), str(output2))
            pages_after_second = json.loads(output2.read_text())

        assert len(pages_after_second) == len(pages_after_first), \
            "C-07 NOT FIXED: pages accumulated across runs"

    def test_empty_pages_skipped(
        self, processor_with_client, valid_pdf, mock_fitz_doc, allowed_base
    ):
        """M-06 fix: blank OCR pages should be skipped, not crash."""
        # Make the mock return empty string for OCR
        processor_with_client._client.ocr.process.return_value.pages[0].markdown = ""
        output = allowed_base / "TESTCO" / "2024" / "out.json"
        output.parent.mkdir(parents=True, exist_ok=True)

        with patch("codebase.ocrprocessor.ocrprocessor.fitz.open",
                   return_value=mock_fitz_doc):
            # Should not raise — blank pages are valid
            processor_with_client._process_pages(str(valid_pdf), str(output))


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:

    def test_run_returns_output_file_path(
        self, processor_with_client, valid_pdf, mock_fitz_doc, allowed_base
    ):
        """C-02 fix: run() must accept and return output_file."""
        output = str(allowed_base / "TESTCO" / "2024" / "result.json")
        (allowed_base / "TESTCO" / "2024").mkdir(parents=True, exist_ok=True)

        with patch("codebase.ocrprocessor.ocrprocessor.fitz.open",
                   return_value=mock_fitz_doc):
            result = processor_with_client.run(
                pdf_path=str(valid_pdf),
                output_file=output,
            )
        assert result == output