# tests/integration/test_pipeline.py
"""
Integration tests: OCRProcessor + validator running together.
Uses a real minimal PDF (tests/fixtures/minimal.pdf) but mocks the Mistral API.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from codebase.ocrprocessor.ocrprocessor import OCRProcessor

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "minimal.pdf"


@pytest.fixture(autouse=True)
def patch_health_check(monkeypatch):
    monkeypatch.setattr(
        "codebase.ocrprocessor.ocrprocessor.assert_system_health",
        lambda **kwargs: None,
    )


class TestFullPipeline:

    def test_clean_ocr_output_produces_valid_json(self, allowed_base):
        """End-to-end: PDF → mock OCR → validate → write JSON."""
        if not FIXTURE_PDF.exists():
            pytest.skip("tests/fixtures/minimal.pdf not found — run creation script first")

        output = allowed_base / "TESTCO" / "2024" / "TESTCO_ANNUAL_2024.json"
        output.parent.mkdir(parents=True)

        mock_page = MagicMock()
        mock_page.markdown = "Revenue: ₹1,000 Crores. Profit: ₹100 Crores."
        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        # Also need to copy the PDF into the allowed_base structure
        import shutil
        src_pdf = allowed_base / "TESTCO" / "2024" / "TESTCO_ANNUAL_2024.pdf"
        shutil.copy(FIXTURE_PDF, src_pdf)

        with patch("codebase.ocrprocessor.ocrprocessor.Mistral") as mock_cls, \
             patch("codebase.ocrprocessor.ocrprocessor.get_logger",
                   return_value=MagicMock()):
            client = MagicMock()
            client.ocr.process.return_value = mock_response
            mock_cls.return_value = client

            proc = OCRProcessor(api_key="test-key")
            result = proc.run(pdf_path=str(src_pdf), output_file=str(output))

        assert result == str(output)
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data) >= 1
        assert "page_number" in data[0]

    def test_injection_in_ocr_output_aborts_pipeline(self, allowed_base):
        """If Mistral returns injected text, the pipeline must reject it."""
        output = allowed_base / "TESTCO" / "2024" / "TESTCO_ANNUAL_2024.json"
        output.parent.mkdir(parents=True)

        if not FIXTURE_PDF.exists():
            pytest.skip("Fixture PDF not found")

        import shutil
        src_pdf = allowed_base / "TESTCO" / "2024" / "TESTCO_ANNUAL_2024.pdf"
        shutil.copy(FIXTURE_PDF, src_pdf)

        mock_page = MagicMock()
        mock_page.markdown = "'; DROP TABLE financials;--"  # SQL injection
        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        with patch("codebase.ocrprocessor.ocrprocessor.Mistral") as mock_cls, \
             patch("codebase.ocrprocessor.ocrprocessor.get_logger",
                   return_value=MagicMock()):
            client = MagicMock()
            client.ocr.process.return_value = mock_response
            mock_cls.return_value = client

            proc = OCRProcessor(api_key="test-key")
            with pytest.raises(ValueError, match="SQL"):
                proc.run(pdf_path=str(src_pdf), output_file=str(output))