# =============================================================================
# CELL 6 — Input Validator
# =============================================================================
"""Central validation and sanitisation for public pipeline inputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from config import CONFIG
from errorhandler import (
    FileTooLargeError,
    InvalidSchemaError,
    PathSafetyError,
    PromptInjectionError,
    UnsafeQueryFilterError,
    ValidationError,
)
from logger import get_logger


class InputValidator:
    """Static validation methods for all pipeline inputs."""

    _INJECTION_PATTERNS = re.compile(
        r"(ignore\s+(all|any|previous)|disregard\s+(all|any|previous)|system\s+prompt|"
        r"developer\s+message|you\s+are\s+now|pretend\s+you|act\s+as|jailbreak|"
        r"reveal\s+(the\s+)?(prompt|instructions|secrets)|exfiltrate|tool\s*call|"
        r"^\s*(system|assistant|developer|tool)\s*:)",
        re.IGNORECASE | re.MULTILINE,
    )
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _COLLECTION_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
    _SCRIP_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")
    _ALLOWED_DOC_TYPES = {"ANNUAL_REPORT", "ANNUAL", "QUARTERLY", "TRANSCRIPT", "PRESENTATION"}
    _ALLOWED_CHROMA_FILTER_KEYS = {
        "company",
        "financial_year",
        "report_year",
        "year",
        "doc_type",
        "source",
        "file_name",
        "document_name",
        "page_number",
        "page",
        "page_no",
        "parent_id",
    }
    _ALLOWED_CHROMA_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$and", "$or"}

    @staticmethod
    def validate_question(question: str) -> str:
        """Validate and sanitise a user question."""
        logger = get_logger("validator")
        if not isinstance(question, str):
            logger.error("Question is not a string", event="validation_failed", type=type(question).__name__)
            raise ValidationError("Question must be a string.")
        question = InputValidator._CONTROL_CHARS.sub("", question).strip()
        if len(question) < CONFIG.MIN_QUESTION_LENGTH:
            raise ValidationError(f"Question too short (min {CONFIG.MIN_QUESTION_LENGTH} chars).")
        if len(question) > CONFIG.MAX_QUESTION_LENGTH:
            raise ValidationError(f"Question too long (max {CONFIG.MAX_QUESTION_LENGTH} chars).")
        InputValidator.detect_prompt_injection(question, field="question")
        return question

    @staticmethod
    def detect_prompt_injection(text: str, *, field: str = "text", strict: bool = True) -> bool:
        """Detect common prompt-injection patterns in trusted inputs or documents."""
        if not isinstance(text, str):
            return False
        if InputValidator._INJECTION_PATTERNS.search(text):
            get_logger("validator").warning(
                "Possible prompt injection detected",
                event="prompt_injection_detected",
                field=field,
                preview=text[:160],
            )
            if strict:
                raise PromptInjectionError(f"{field} contains disallowed prompt-injection patterns.")
            return True
        return False

    @staticmethod
    def validate_scrip(scrip: str) -> str:
        """Validate a stock scrip symbol."""
        if not isinstance(scrip, str):
            raise ValidationError("Scrip must be a string.")
        scrip = scrip.strip().upper()
        if not InputValidator._SCRIP_RE.match(scrip):
            raise ValidationError(f"Invalid scrip format: '{scrip}'. Expected 1-20 alphanumeric/&/- characters.")
        return scrip

    @staticmethod
    def validate_fiscal_year(fy: str | int) -> str:
        """Normalise a fiscal year string/int to FY25 format."""
        fy_str = str(fy).strip().upper()
        m = re.match(r"^(?:FY)?(\d{4})$", fy_str)
        if m:
            return f"FY{m.group(1)[-2:]}"
        m = re.match(r"^(?:FY)?(\d{2})$", fy_str)
        if m:
            return f"FY{m.group(1)}"
        raise ValidationError(f"Cannot parse fiscal year from: '{fy}'")

    @staticmethod
    def validate_year_int(year: int | str) -> int:
        """Validate and normalise a four-digit report year."""
        try:
            value = int(year)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Year must be an integer.") from exc
        if value < 1990 or value > 2100:
            raise ValidationError("Year must be between 1990 and 2100.")
        return value

    @staticmethod
    def validate_doc_type(doc_type: str) -> str:
        """Validate a document type string."""
        if not isinstance(doc_type, str):
            raise ValidationError("Document type must be a string.")
        cleaned = doc_type.strip().upper()
        if cleaned not in InputValidator._ALLOWED_DOC_TYPES:
            raise ValidationError(f"Unsupported document type: '{doc_type}'.")
        return cleaned

    @staticmethod
    def _ensure_within_base(path: Path, base: Path, label: str) -> Path:
        try:
            path.relative_to(base)
        except ValueError as exc:
            get_logger("validator").warning("Path traversal attempt", event="path_traversal_detected", path=str(path), base=str(base))
            raise PathSafetyError(f"{label} is outside the allowed directory.") from exc
        return path

    @staticmethod
    def validate_path(path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str] | None = None, suffixes: set[str] | None = None, must_exist: bool = True, label: str = "Path") -> str:
        """Validate a filesystem path is inside base_dir, has an allowed suffix, and exists if required."""
        if not isinstance(path, (str, os.PathLike)):
            raise ValidationError(f"{label} must be a path string.")
        resolved = Path(path).expanduser().resolve()
        base = Path(base_dir or CONFIG.UPLOADS_PATH).expanduser().resolve()
        InputValidator._ensure_within_base(resolved, base, label)
        if suffixes and resolved.suffix.lower() not in suffixes:
            raise ValidationError(f"{label} must have one of these extensions: {sorted(suffixes)}")
        if must_exist and not resolved.is_file():
            raise ValidationError(f"{label} not found: '{resolved}'")
        return str(resolved)

    @staticmethod
    def validate_pdf_path(path: str) -> str:
        """Validate a PDF file path for safety, accessibility, and max size."""
        resolved = InputValidator.validate_path(path, suffixes={".pdf"}, must_exist=True, label="PDF path")
        size_mb = os.path.getsize(resolved) / (1024 * 1024)
        if size_mb > CONFIG.MAX_PDF_SIZE_MB:
            raise FileTooLargeError(f"PDF too large ({size_mb:.1f} MB). Max allowed: {CONFIG.MAX_PDF_SIZE_MB} MB.")
        return resolved

    @staticmethod
    def validate_json_path(path: str, *, must_exist: bool = True) -> str:
        """Validate a JSON artifact path inside uploads."""
        return InputValidator.validate_path(path, suffixes={".json"}, must_exist=must_exist, label="JSON path")

    @staticmethod
    def validate_output_path(path: str, *, suffixes: set[str] | None = None) -> str:
        """Validate an output artifact path inside uploads."""
        return InputValidator.validate_path(path, suffixes=suffixes or {".json"}, must_exist=False, label="Output path")

    @staticmethod
    def validate_top_k(top_k: int | None, *, default: int, max_value: int | None = None) -> int:
        """Validate retrieval result limit."""
        value = default if top_k is None else top_k
        if not isinstance(value, int):
            raise ValidationError("top_k must be an integer.")
        hard_max = max_value or max(CONFIG.FINAL_TOP_K, CONFIG.SEMANTIC_TOP_K, 1)
        if value < 1 or value > hard_max:
            raise ValidationError(f"top_k must be between 1 and {hard_max}.")
        return value

    @staticmethod
    def validate_collection_name(collection_name: str) -> str:
        """Validate a Chroma collection name."""
        if not isinstance(collection_name, str) or not InputValidator._COLLECTION_RE.match(collection_name):
            raise ValidationError("Invalid Chroma collection name.")
        return collection_name

    @staticmethod
    def validate_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate Chroma metadata filters against allowed keys/operators."""
        if where is None:
            return None
        if not isinstance(where, dict):
            raise UnsafeQueryFilterError("Chroma where filter must be a dictionary.")

        def walk(obj: Any, depth: int = 0) -> None:
            if depth > 3:
                raise UnsafeQueryFilterError("Chroma where filter is too deeply nested.")
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith("$"):
                        if key not in InputValidator._ALLOWED_CHROMA_OPERATORS:
                            raise UnsafeQueryFilterError(f"Unsupported Chroma operator: {key}")
                    elif key not in InputValidator._ALLOWED_CHROMA_FILTER_KEYS:
                        raise UnsafeQueryFilterError(f"Unsupported Chroma filter key: {key}")
                    walk(value, depth + 1)
            elif isinstance(obj, list):
                if len(obj) > 50:
                    raise UnsafeQueryFilterError("Chroma filter list is too large.")
                for item in obj:
                    walk(item, depth + 1)
            elif obj is not None and not isinstance(obj, (str, int, float, bool)):
                raise UnsafeQueryFilterError("Chroma filter values must be scalar/list/dict.")

        walk(where)
        return where

    @staticmethod
    def validate_query_texts(query_texts: list[str]) -> list[str]:
        """Validate Chroma/RAG query texts."""
        if not isinstance(query_texts, list) or not query_texts:
            raise ValidationError("query_texts must be a non-empty list.")
        if len(query_texts) > 5:
            raise ValidationError("Too many query texts.")
        return [InputValidator.validate_question(q) for q in query_texts]

    @staticmethod
    def validate_raw_ocr_pages(pages: Any) -> list[dict[str, Any]]:
        """Validate raw OCR JSON page list."""
        if not isinstance(pages, list):
            raise InvalidSchemaError("OCR JSON must be a list of page objects.")
        for idx, page in enumerate(pages):
            if not isinstance(page, dict):
                raise InvalidSchemaError(f"Page {idx} must be an object.")
            if "text" not in page or not isinstance(page["text"], str):
                raise InvalidSchemaError(f"Page {idx} is missing text.")
            if "page_num" not in page and "page_number" not in page:
                raise InvalidSchemaError(f"Page {idx} is missing page number.")
            InputValidator.detect_prompt_injection(page["text"], field=f"page[{idx}].text", strict=False)
        return pages

    @staticmethod
    def validate_embedding_payload(payload: Any) -> Any:
        """Validate embedding-ready records before Chroma upsert."""
        record_sets: list[tuple[str, list[Any]]]
        if isinstance(payload, dict) and "parents" in payload and "children" in payload:
            record_sets = [("parents", payload.get("parents", [])), ("children", payload.get("children", []))]
        elif isinstance(payload, list):
            record_sets = [("records", payload)]
        else:
            raise InvalidSchemaError("Embedding JSON must be a record list or parent/child bundle.")
        for name, records in record_sets:
            if not isinstance(records, list):
                raise InvalidSchemaError(f"{name} must be a list.")
            seen: set[str] = set()
            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    raise InvalidSchemaError(f"{name}[{idx}] must be an object.")
                if not isinstance(rec.get("id"), str) or not rec["id"]:
                    raise InvalidSchemaError(f"{name}[{idx}] is missing id.")
                if rec["id"] in seen:
                    raise InvalidSchemaError(f"Duplicate id in {name}: {rec['id']}")
                seen.add(rec["id"])
                if not isinstance(rec.get("text"), str) or not rec["text"].strip():
                    raise InvalidSchemaError(f"{name}[{idx}] is missing text.")
                if rec.get("metadata") is not None and not isinstance(rec.get("metadata"), dict):
                    raise InvalidSchemaError(f"{name}[{idx}].metadata must be an object.")
        return payload

    @staticmethod
    def load_json_file(path: str) -> Any:
        """Load JSON from a validated path."""
        json_path = InputValidator.validate_json_path(path, must_exist=True)
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise InvalidSchemaError(f"Invalid JSON: {json_path}") from exc

    @staticmethod
    def sanitize_untrusted_context(text: str, *, max_chars: int = 12_000) -> str:
        """Wrap retrieved/document text so prompt templates treat it as data, not instructions."""
        if not isinstance(text, str):
            raise ValidationError("Context text must be a string.")
        cleaned = InputValidator._CONTROL_CHARS.sub("", text).strip()
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "\n[TRUNCATED_CONTEXT]"
        return f"<untrusted_context>\n{cleaned}\n</untrusted_context>"

    @staticmethod
    def prompt_injection_score(text: str) -> int:
        """Return a simple rule-based suspicion score for document/query text."""
        if not isinstance(text, str) or not text:
            return 0
        score = 0
        if InputValidator._INJECTION_PATTERNS.search(text):
            score += 5
        lowered = text.lower()
        score += sum(1 for marker in ("ignore", "system", "developer", "secret", "tool", "instruction") if marker in lowered)
        if "```" in text or "<!--" in text:
            score += 1
        return score

    @staticmethod
    def validate_llm_answer(answer: str, source_chunks: list[Any]) -> str:
        """Validate LLM output is non-empty and not obviously ungrounded/unsafe."""
        if not isinstance(answer, str) or not answer.strip():
            raise ValidationError("LLM answer is empty.")
        cleaned = InputValidator._CONTROL_CHARS.sub("", answer).strip()
        InputValidator.detect_prompt_injection(cleaned, field="llm_answer", strict=True)
        if not source_chunks:
            lowered = cleaned.lower()
            if "not able to" not in lowered and "insufficient" not in lowered and "could not find" not in lowered:
                raise ValidationError("LLM answer has no supporting source chunks.")
        return cleaned

    @staticmethod
    def validate_chunk_count(count: int, context: str = "") -> None:
        """Warn if chunk count is outside the expected range."""
        logger = get_logger("validator")
        if count < CONFIG.CHUNK_COUNT_MIN:
            logger.warning("Chunk count below expected minimum", event="chunk_count_low", count=count, min=CONFIG.CHUNK_COUNT_MIN, context=context)
        elif count > CONFIG.CHUNK_COUNT_MAX:
            logger.warning("Chunk count above expected maximum", event="chunk_count_high", count=count, max=CONFIG.CHUNK_COUNT_MAX, context=context)
