"""
mistral_client.py — Mistral OCR API client.

Owns two things only:
  1. Building the authenticated Mistral client.
  2. Sending one page image to the OCR API and returning the text.

Nothing about PDFs, file paths, or JSON lives here.
"""
from __future__ import annotations

import base64

from mistralai.client import Mistral


class MistralClient:
    """
    Wraps the Mistral OCR API for a single-page call.

    The API key is used only to construct the client and is not
    stored on the instance after __init__ completes.

    Args:
        api_key: Mistral API key.
        model:   OCR model identifier (from CONFIG.MISTRAL_MODEL_OCR).
    """

    DEFAULT_TIMEOUT: int = 30   # seconds — prevents hung connections

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Mistral(api_key=api_key, timeout=self.DEFAULT_TIMEOUT)
        self._model  = model

    def process_page(self, image_bytes: bytes) -> str:
        """
        Send a PNG image to the Mistral OCR API and return the markdown text.

        Args:
            image_bytes: Raw PNG bytes for a single rendered PDF page.

        Returns:
            Markdown-formatted OCR text. Empty string if no text is detected.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = self._client.ocr.process(
            model=self._model,
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{b64}",
            },
        )
        return response.pages[0].markdown if response.pages else ""