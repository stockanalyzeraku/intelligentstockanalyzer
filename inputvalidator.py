# =============================================================================
# CELL 6 — Input Validator
# =============================================================================
"""
InputValidator: static methods for validating every external input
before it touches the processing pipeline.
"""

import os
import re
from typing import Optional
from logger import get_logger
import config

class ValidationError(ValueError):
    """Raised when input validation fails."""


class InputValidator:
    """
    Static validation methods for all pipeline inputs.

    All methods return the (possibly normalised) input on success
    and raise ValidationError on failure.
    """

    _INJECTION_PATTERNS = re.compile(
        r"(ignore previous|ignore all|disregard|system prompt|"
        r"you are now|pretend you|act as|jailbreak)",
        re.IGNORECASE,
    )

    def __init__(self):
        self._CONFIG = config.Config.get_instance()

    
    def validate_question(self, question: str) -> str:
        """
        Validate and sanitise a user question.

        Parameters
        ----------
        question : str
            Raw question from the user.

        Returns
        -------
        str
            Stripped question.

        Raises
        ------
        ValidationError
            If question is empty, too short, too long, or contains
            prompt-injection patterns.
        """
        _v_logger = get_logger("validator")
        if not isinstance(question, str):
            _v_logger.error("Question is not a string", type=type(question).__name__)
            raise ValidationError("Question must be a string.")
        question = question.strip()
        if len(question) < self._CONFIG.MIN_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too short (min {self._CONFIG.MIN_QUESTION_LENGTH} chars)."
            )
        if len(question) > self._CONFIG.MAX_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too long (max {self._CONFIG.MAX_QUESTION_LENGTH} chars)."
            )
        if InputValidator._INJECTION_PATTERNS.search(question):
            _v_logger.warning("Possible prompt injection detected", question=question[:80])
            raise ValidationError("Question contains disallowed patterns.")
        return question

    
    def validate_scrip(scrip: str) -> str:
        """
        Validate a stock scrip symbol (e.g. 'RELIANCE', 'TCS').

        Parameters
        ----------
        scrip : str
            Scrip symbol to validate.

        Returns
        -------
        str
            Upper-cased scrip.

        Raises
        ------
        ValidationError
            If scrip format is invalid.
        """
        if not isinstance(scrip, str):
            raise ValidationError("Scrip must be a string.")
        scrip = scrip.strip().upper()
        if not re.match(r"^[A-Z0-9&\-]{1,20}$", scrip):
            raise ValidationError(
                f"Invalid scrip format: '{scrip}'. "
                "Expected 1-20 alphanumeric/&/- characters."
            )
        return scrip

    
    def validate_fiscal_year(fy: str) -> str:
        """
        Normalise a fiscal year string to FY25 format.

        Accepts: 'FY2025', '2025', 'fy25', 'FY25', '25'.

        Parameters
        ----------
        fy : str
            Fiscal year in any accepted format.

        Returns
        -------
        str
            Normalised as 'FY25' (two-digit year suffix).

        Raises
        ------
        ValidationError
            If the string cannot be interpreted as a fiscal year.
        """
        if not isinstance(fy, str):
            raise ValidationError("Fiscal year must be a string.")
        fy = fy.strip().upper()
        # FY2025 or 2025
        m = re.match(r"^(?:FY)?(\d{4})$", fy)
        if m:
            return f"FY{m.group(1)[-2:]}"
        # FY25 or 25
        m = re.match(r"^(?:FY)?(\d{2})$", fy)
        if m:
            return f"FY{m.group(1)}"
        raise ValidationError(f"Cannot parse fiscal year from: '{fy}'")

    
    def validate_pdf_path(self, path: str) -> str:
        """
        Validate a PDF file path for safety and accessibility.

        Parameters
        ----------
        path : str
            Absolute or relative path to a PDF file.

        Returns
        -------
        str
            Resolved absolute path.

        Raises
        ------
        ValidationError
            If path traversal detected, extension wrong, file missing,
            or file exceeds MAX_PDF_SIZE_MB.
        """
        _v_logger = get_logger("validator")
        if not isinstance(path, str):
            raise ValidationError("PDF path must be a string.")
        abs_path = os.path.realpath(path)
        # Path traversal check
        uploads_real = os.path.realpath(self._CONFIG.UPLOADS_PATH)
        if not abs_path.startswith(uploads_real):
            _v_logger.warning("Path traversal attempt", path=path)
            raise ValidationError(
                f"Path '{path}' is outside the allowed uploads directory."
            )
        if not abs_path.lower().endswith(".pdf"):
            raise ValidationError(f"File must have a .pdf extension: '{path}'")
        if not os.path.isfile(abs_path):
            raise ValidationError(f"File not found: '{abs_path}'")
        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        if size_mb > self._CONFIG.MAX_PDF_SIZE_MB:
            raise ValidationError(
                f"PDF too large ({size_mb:.1f} MB). Max allowed: {self._CONFIG.MAX_PDF_SIZE_MB} MB."
            )
        return abs_path

    def validate_chunk_count(self, count: int, context: str = "") -> None:
        """
        Warn if chunk count is outside the expected range.

        Parameters
        ----------
        count : int
            Number of chunks produced.
        context : str
            Description of what was chunked (for log messages).
        """
        _v_logger = get_logger("validator")
        if count < self._CONFIG.CHUNK_COUNT_MIN:
            _v_logger.warning(
                "Chunk count below expected minimum",
                count=count,
                min=self._CONFIG.CHUNK_COUNT_MIN,
                context=context,
            )
        elif count > self._CONFIG.CHUNK_COUNT_MAX:
            _v_logger.warning(
                "Chunk count above expected maximum",
                count=count,
                max=self._CONFIG.CHUNK_COUNT_MAX,
                context=context,
            )


_val_logger = get_logger("validator")
_val_logger.info("InputValidator ready.")

# ----------------------------------------------------------------------------
# Cell 6: Input Validator
# Purpose: Validate and sanitise all pipeline inputs before processing.
# Key Classes: InputValidator, ValidationError
# Key Functions:
#   InputValidator.validate_question(question) → str
#   InputValidator.validate_scrip(scrip) → str
#   InputValidator.validate_fiscal_year(fy) → str
#   InputValidator.validate_pdf_path(path) → str
#   InputValidator.validate_chunk_count(count, context) → None
# Key Constants/Config: CONFIG.MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH,
#   MAX_PDF_SIZE_MB, CHUNK_COUNT_MIN/MAX, UPLOADS_PATH
# Imports exported: InputValidator, ValidationError
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: validate_pdf_path does a real-path check against
#   CONFIG.UPLOADS_PATH — files outside that tree are rejected.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
