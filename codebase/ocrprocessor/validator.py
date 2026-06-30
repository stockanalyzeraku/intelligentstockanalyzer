"""
validator.py — every input that enters the ocrprocessor is validated here.

Sections
────────
1.  Path validation          _validate_filepath, _validate_output_path
2.  API / client validation  _validate_api_key, _validate_model_name
3.  Rendering validation     _validate_dpi, _validate_image_bytes
4.  OCR text validation      _check_dangerous_format, _check_pathological_nesting,
                              _check_markdown_urls, _validate_ocr_text
5.  Page / database safety   _validate_page_number, _validate_page_content,
                              _validate_pages_list
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

from codebase.ocrprocessor.exceptions import FilePathError
from codebase.ocrprocessor.skelton import (
    ALLOWED_BASE,
    ALLOWED_EXTENSIONS,
    MAX_TEXT_LENGTH,
    CODE_INJECTION_PATTERNS,
    SQL_INJECTION_PATTERNS,
    PATH_TRAVERSAL_PATTERN,
    DANGEROUS_DATA_URI_PATTERN,
    MIN_YEAR,
    MAX_YEAR,
    NESTING_THRESHOLD,
    DANGEROUS_URI_SCHEMES,
    MARKDOWN_URL_PATTERN,
    DANGEROUS_FORMAT_CHARS,
    ZERO_WIDTH_CHARS,
    PRIVATE_HOSTS,
    PRIVATE_IP_PATTERN,
    _SCRIPT_NAME_RE,
    BIDI_OVERRIDE_CHARS,
    DIRECTIONAL_MARKS,
    SEPARATOR_INJECTION_CHARS,
    TAG_BLOCK_RANGE,
    INTERLINEAR_CHARS,
    OBJECT_REPLACEMENT_CHAR,
    MAX_REPLACEMENT_CHAR_DENSITY,
    PageContent,
)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Rendering ────────────────────────────────────────────────────────────────
_PNG_MAGIC       = b"\x89PNG\r\n\x1a\n"  # first 8 bytes of every valid PNG file
_MIN_IMAGE_BYTES = 1_000                  # 1 KB  — below this the render is corrupt
_MAX_IMAGE_BYTES = 50_000_000             # 50 MB — a single page should never exceed this
_MIN_DPI         = 72                     # screen resolution minimum
_MAX_DPI         = 600                    # high-quality scan maximum

# ── API / client ─────────────────────────────────────────────────────────────
_MIN_API_KEY_LEN    = 20
_MAX_API_KEY_LEN    = 500
_MAX_MODEL_NAME_LEN = 200
_API_KEY_RE         = re.compile(r"^[A-Za-z0-9\-_.]+$")
_MODEL_NAME_RE      = re.compile(r"^[A-Za-z0-9\-_./]+$")

# ── Page / database ───────────────────────────────────────────────────────────
# Must stay in sync with MAX_PDF_PAGES in pdfrenderer.py.
_MAX_PDF_PAGES   = 500
_MIN_PAGE_NUMBER = 1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PATH VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _validate_filepath(path) -> Path:
    """
    Validate an input file path end-to-end.

    Checks (in order):
        1. Not None / not empty
        2. Resolves inside ALLOWED_BASE   (path traversal prevention)
        3. File exists and is a regular file
        4. Directory structure: uploads/<script>/<year>/<type>/<filename>
        5. Script name matches [A-Z0-9_-]{1,64}
        6. Year is a 4-digit integer in [MIN_YEAR, MAX_YEAR]
        7. Extension is in ALLOWED_EXTENSIONS
        8. File is not empty (0 bytes)
        9. Content-level: PDF has correct header; JSON is parseable UTF-8

    Returns:
        Resolved absolute Path.

    Raises:
        FilePathError: path-level violations (empty, traversal, missing).
        ValueError:    structural or content-level violations.
    """
    if path is None or str(path).strip() == "":
        raise FilePathError(path, "File path is empty")

    resolved = Path(path).expanduser().resolve()

    # ── 1. Containment check ─────────────────────────────────────────────────
    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(
            path,
            "Path is not inside the allowed base directory "
            "(possible path traversal attempt)",
        )

    # ── 2. Existence and type check ──────────────────────────────────────────
    if not resolved.exists():
        raise FilePathError(path, f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise FilePathError(path, f"Path is not a regular file: {resolved}")

    # ── 3. Directory structure ────────────────────────────────────────────────
    try:
        relative_parts = resolved.relative_to(ALLOWED_BASE).parts
    except ValueError:
        raise ValueError(f"Could not compute relative path for: {resolved}")

    if len(relative_parts) != 4:
        raise ValueError(
            f"Expected structure 'uploads/<script>/<year>/<type>/<filename>', "
            f"got {len(relative_parts)} segment(s): {relative_parts}"
        )

    script_name, year_str, _file_type, _filename = relative_parts

    # ── 4. Script-name segment ───────────────────────────────────────────────
    if not script_name or not script_name.strip():
        raise ValueError("Script/folder name segment is empty")
    if script_name.startswith("."):
        raise ValueError(
            f"Script folder name looks hidden/invalid: '{script_name}'"
        )
    if not _SCRIPT_NAME_RE.match(script_name.upper()):
        raise ValueError(
            f"Script name contains disallowed characters or exceeds 64 chars: "
            f"'{script_name}'"
        )

    # ── 5. Year segment ──────────────────────────────────────────────────────
    if not year_str.isdigit():
        raise ValueError(f"Year segment is not numeric: '{year_str}'")
    year = int(year_str)
    if not (MIN_YEAR <= year <= MAX_YEAR):
        raise ValueError(
            f"Year {year} is out of plausible range ({MIN_YEAR}–{MAX_YEAR})"
        )

    # ── 6. Extension ─────────────────────────────────────────────────────────
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{suffix}'; "
            f"expected one of {sorted(ALLOWED_EXTENSIONS)}"
        )

    # ── 7. Non-empty ─────────────────────────────────────────────────────────
    if resolved.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {resolved}")

    # ── 8. Content-level ─────────────────────────────────────────────────────
    if suffix == ".json":
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"File is not valid JSON: {exc}")
        except UnicodeDecodeError:
            raise ValueError("JSON file is not valid UTF-8 text")

    elif suffix == ".pdf":
        with open(resolved, "rb") as fh:
            header = fh.read(5)
        if header != b"%PDF-":
            raise ValueError(
                "File does not have a valid PDF header (expected '%PDF-')"
            )

    return resolved


def _validate_output_path(output_path: str) -> Path:
    """
    Validate a write-destination path. The file need not exist yet.

    Checks:
        1. Not None / not empty
        2. Resolves inside ALLOWED_BASE
        3. Extension is in ALLOWED_EXTENSIONS
        4. Parent directory exists or can be created

    Returns:
        Resolved absolute Path.

    Raises:
        FilePathError: on any violation.
    """
    if not output_path or str(output_path).strip() == "":
        raise FilePathError(output_path, "Output path is empty")

    resolved = Path(output_path).expanduser().resolve()

    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(
            resolved, "Output path is outside the allowed base directory"
        )

    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise FilePathError(
            resolved,
            f"Output extension '{resolved.suffix}' is not allowed; "
            f"expected one of {sorted(ALLOWED_EXTENSIONS)}",
        )

    if not resolved.parent.exists():
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FilePathError(resolved, f"Cannot create output directory: {exc}")

    return resolved


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — API / CLIENT VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _validate_api_key(api_key: str) -> str:
    """
    Validate a Mistral (or equivalent) API key before it is used to build a client.

    Checks:
        1. Not None
        2. Is a string
        3. Not empty / not whitespace-only
        4. No leading or trailing whitespace
        5. Length within [_MIN_API_KEY_LEN, _MAX_API_KEY_LEN]
        6. No embedded whitespace (space, tab, newline, carriage return)
        7. Characters restricted to alphanumeric, hyphen, underscore, dot

    Returns:
        The validated API key unchanged.

    Raises:
        ValueError: on any violation.
    """
    if api_key is None:
        raise ValueError("API key must not be None")
    if not isinstance(api_key, str):
        raise ValueError(
            f"API key must be a string, got {type(api_key).__name__}"
        )
    if not api_key:
        raise ValueError("API key is an empty string")
    if api_key != api_key.strip():
        raise ValueError(
            "API key has leading or trailing whitespace — "
            "check your environment variable or config file"
        )
    if len(api_key) < _MIN_API_KEY_LEN:
        raise ValueError(
            f"API key is too short ({len(api_key)} chars); "
            f"minimum is {_MIN_API_KEY_LEN}"
        )
    if len(api_key) > _MAX_API_KEY_LEN:
        raise ValueError(
            f"API key is too long ({len(api_key)} chars); "
            f"maximum is {_MAX_API_KEY_LEN}"
        )
    for bad_char in (" ", "\t", "\n", "\r"):
        if bad_char in api_key:
            raise ValueError(
                "API key contains embedded whitespace — "
                "keys must not contain spaces, tabs, or newlines"
            )
    if not _API_KEY_RE.match(api_key):
        raise ValueError(
            "API key contains disallowed characters. "
            "Only alphanumeric characters, hyphens, underscores, "
            "and dots are permitted."
        )
    return api_key


def _validate_model_name(model: str) -> str:
    """
    Validate an OCR model name before it is sent to the API.

    Prevents injection of unexpected values into the API request body.

    Checks:
        1. Not None
        2. Is a string
        3. Not empty / not whitespace-only
        4. Length within [1, _MAX_MODEL_NAME_LEN]
        5. Characters restricted to alphanumeric, hyphen, underscore,
           dot, and forward-slash (covers versioned paths like mistral/v2)
        6. No whitespace anywhere

    Returns:
        The validated model name, stripped of surrounding whitespace.

    Raises:
        ValueError: on any violation.
    """
    if model is None:
        raise ValueError("Model name must not be None")
    if not isinstance(model, str):
        raise ValueError(
            f"Model name must be a string, got {type(model).__name__}"
        )
    stripped = model.strip()
    if not stripped:
        raise ValueError("Model name is empty or whitespace-only")
    if len(stripped) > _MAX_MODEL_NAME_LEN:
        raise ValueError(
            f"Model name is too long ({len(stripped)} chars); "
            f"maximum is {_MAX_MODEL_NAME_LEN}"
        )
    if not _MODEL_NAME_RE.match(stripped):
        raise ValueError(
            f"Model name '{stripped}' contains disallowed characters. "
            "Only alphanumeric characters, hyphens, underscores, "
            "dots, and forward-slashes are permitted."
        )
    return stripped


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RENDERING VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _validate_dpi(dpi: int) -> int:
    """
    Validate a DPI value before PDF rendering.

    DPI controls both OCR accuracy and memory use:
        - Too low  → text too blurry for the OCR model to read
        - Too high → single page image can exceed hundreds of MB

    Allowed range: [_MIN_DPI, _MAX_DPI]  (72 – 600).

    Returns:
        The validated DPI value unchanged.

    Raises:
        TypeError:  if dpi is not an integer.
        ValueError: if dpi is outside the allowed range.
    """
    if isinstance(dpi, bool):
        raise TypeError("DPI must not be a boolean")
    if not isinstance(dpi, int):
        raise TypeError(
            f"DPI must be an integer, got {type(dpi).__name__}"
        )
    if dpi <= 0:
        raise ValueError(f"DPI must be a positive integer, got {dpi}")
    if dpi < _MIN_DPI:
        raise ValueError(
            f"DPI {dpi} is below the minimum of {_MIN_DPI}. "
            "OCR accuracy degrades severely at low resolution."
        )
    if dpi > _MAX_DPI:
        raise ValueError(
            f"DPI {dpi} exceeds the maximum of {_MAX_DPI}. "
            "High DPI values cause excessive memory use per page."
        )
    return dpi


def _validate_image_bytes(image_bytes: bytes, page_number: int) -> bytes:
    """
    Validate raw PNG bytes produced by the PDF renderer before sending to OCR.

    Catches corrupt renders before an API call is made.

    Checks:
        1. Not None
        2. Is bytes type
        3. Not empty
        4. Size >= _MIN_IMAGE_BYTES (1 KB) — guards against corrupt renders
        5. Size <= _MAX_IMAGE_BYTES (50 MB) — guards against memory exhaustion
        6. Starts with the PNG magic bytes — confirms correct image format

    Args:
        image_bytes:  Raw bytes from PDFRenderer.render_pages().
        page_number:  1-indexed page number — included in error messages.

    Returns:
        The validated bytes unchanged.

    Raises:
        TypeError:  if image_bytes is not bytes.
        ValueError: on any size or format violation.
    """
    if image_bytes is None:
        raise ValueError(f"Page {page_number}: image bytes must not be None")
    if not isinstance(image_bytes, bytes):
        raise TypeError(
            f"Page {page_number}: image bytes must be of type bytes, "
            f"got {type(image_bytes).__name__}"
        )

    size = len(image_bytes)

    if size == 0:
        raise ValueError(
            f"Page {page_number}: image bytes are empty — "
            "PDF renderer produced no output"
        )
    if size < _MIN_IMAGE_BYTES:
        raise ValueError(
            f"Page {page_number}: image is suspiciously small ({size} bytes); "
            f"minimum expected is {_MIN_IMAGE_BYTES} bytes. "
            "The render may be corrupt."
        )
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Page {page_number}: image is too large ({size:,} bytes); "
            f"maximum allowed is {_MAX_IMAGE_BYTES:,} bytes. "
            "Reduce DPI or check for a runaway render."
        )
    if not image_bytes.startswith(_PNG_MAGIC):
        raise ValueError(
            f"Page {page_number}: image bytes do not start with the PNG magic header "
            f"(expected {_PNG_MAGIC!r}). "
            "The renderer must produce PNG output."
        )
    return image_bytes


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OCR TEXT VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _check_dangerous_format(text: str) -> None:
    """
    Detect Unicode-based visual spoofing, steganography and injection attacks.

    Must be called on NFC-normalised text BEFORE zero-width stripping so
    that both the raw and stripped forms are inspectable.

    Attack vectors covered:
        - Trojan Source (CVE-2021-42574): Bidi overrides visually reverse text.
        - Directional mark injection: invisible marks alter render direction.
        - JS string injection: U+2028/U+2029 (Zl/Zp) bypass Cc/Cf sweeps.
        - Tag block steganography: U+E0000-U+E007F, deprecated since Unicode 6.0.
        - Interlinear annotation abuse.
        - Object replacement flooding: masks binary data lengths.
        - Zero-width keyword splitting: evasion of pattern detectors.

    Raises:
        ValueError: naming the specific attack and offending codepoints.
    """
    # ── 1. Bidi override / isolate — Trojan Source (CVE-2021-42574) ──────────
    found_bidi = [c for c in text if c in BIDI_OVERRIDE_CHARS]
    if found_bidi:
        raise ValueError(
            "Bidi override/isolate character detected "
            "(Trojan Source attack — CVE-2021-42574). "
            "These characters visually reverse text to hide malicious content. "
            f"Offending codepoints: {[f'U+{ord(c):04X}' for c in found_bidi]}"
        )

    # ── 2. Invisible directional marks ───────────────────────────────────────
    found_dir = [c for c in text if c in DIRECTIONAL_MARKS]
    if found_dir:
        raise ValueError(
            "Invisible directional mark character detected "
            "(text direction manipulation attack). "
            f"Offending codepoints: {[f'U+{ord(c):04X}' for c in found_dir]}"
        )

    # ── 3. Line / paragraph separator injection ───────────────────────────────
    # Category Zl/Zp — NOT caught by the Cc/Cf sweep. Must be explicit.
    found_sep = [c for c in text if c in SEPARATOR_INJECTION_CHARS]
    if found_sep:
        raise ValueError(
            "Line/paragraph separator character detected "
            "(U+2028/U+2029 can escape JS string literals and "
            "cause silent line breaks inside JSON values). "
            f"Offending codepoints: {[f'U+{ord(c):04X}' for c in found_sep]}"
        )

    # ── 4. Unicode Tag block (U+E0000–U+E007F) — invisible steganography ──────
    lo, hi = TAG_BLOCK_RANGE
    tag_chars = [c for c in text if lo <= ord(c) <= hi]
    if tag_chars:
        raise ValueError(
            "Unicode Tag block character detected "
            "(invisible steganography — deprecated since Unicode 6.0). "
            f"Offending codepoints: {[f'U+{ord(c):05X}' for c in tag_chars]}"
        )

    # ── 5. Interlinear annotation characters ──────────────────────────────────
    found_ia = [c for c in text if c in INTERLINEAR_CHARS]
    if found_ia:
        raise ValueError(
            "Interlinear annotation character detected "
            "(no legitimate use in financial OCR text). "
            f"Offending codepoints: {[f'U+{ord(c):04X}' for c in found_ia]}"
        )

    # ── 6. Object replacement character density ───────────────────────────────
    # U+FFFC is category So — NOT caught by the Cc/Cf sweep.
    obj_count = text.count(OBJECT_REPLACEMENT_CHAR)
    if obj_count > MAX_REPLACEMENT_CHAR_DENSITY:
        raise ValueError(
            f"Excessive object replacement characters: {obj_count} found "
            f"(threshold: {MAX_REPLACEMENT_CHAR_DENSITY}). "
            "Possible binary data embedded inside OCR output."
        )

    # ── 7. Zero-width character run — keyword-splitting evasion ──────────────
    if re.search(f"[{ZERO_WIDTH_CHARS}]{{4,}}", text):
        raise ValueError(
            "Abnormal run of >=4 consecutive zero-width characters detected "
            "(keyword-splitting evasion — used to hide injection patterns "
            "by splitting keywords across invisible character boundaries)."
        )


def _check_pathological_nesting(text: str) -> None:
    """
    Detect nesting bombs in markdown syntax that could crash downstream parsers.

    Raises:
        ValueError: if bracket, parenthesis, or blockquote nesting exceeds
                    NESTING_THRESHOLD.
    """
    if re.search(rf"\[{{{NESTING_THRESHOLD},}}", text):
        raise ValueError(
            f"{NESTING_THRESHOLD}+ consecutive '[' characters detected (nesting bomb)"
        )
    if re.search(rf"\({{{NESTING_THRESHOLD},}}", text):
        raise ValueError(
            f"{NESTING_THRESHOLD}+ consecutive '(' characters detected (nesting bomb)"
        )
    if re.search(rf"^(>\s*){{{NESTING_THRESHOLD},}}", text, re.MULTILINE):
        raise ValueError(
            f"{NESTING_THRESHOLD}+ levels of blockquote nesting detected"
        )


def _check_markdown_urls(text: str) -> None:
    """
    Scan all markdown links and images for dangerous schemes and SSRF targets.

    Uses scan_text (zero-width stripped) so zero-width chars embedded inside
    a URI scheme cannot bypass the scheme check.

    Raises:
        ValueError: if any URL uses a dangerous scheme or targets a private host.
    """
    for match in re.finditer(MARKDOWN_URL_PATTERN, text):
        url    = match.group(1)
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host   = (parsed.hostname or "").lower()

        if scheme in DANGEROUS_URI_SCHEMES:
            raise ValueError(
                f"Disallowed URI scheme '{scheme}:' in markdown link/image: {url}"
            )

        # Numeric IP normalisation — catches decimal (2130706433 = 127.0.0.1),
        # octal (0177.0.0.1), and hex (0x7f000001) encoded private IPs.
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                raise ValueError(
                    f"Markdown link/image targets a private or loopback address "
                    f"(possible SSRF): {url}"
                )
        except ValueError as exc:
            if "SSRF" in str(exc):
                raise

        if host in PRIVATE_HOSTS or re.match(PRIVATE_IP_PATTERN, host):
            raise ValueError(
                f"Markdown link/image targets a private or internal host "
                f"(possible SSRF): {url}"
            )


def _validate_ocr_text(text: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Validate and normalise a single page of raw OCR text.

    Returns empty string for genuinely blank pages — callers must skip those.

    Checks (in order):
        1.  Not None
        2.  Is a string
        3.  Blank page  → return ""
        4.  Length <= max_length
        5.  Valid UTF-8 encoding
        6.  NFC normalisation
        7.  No null bytes
        8.  Unicode spoofing/format attacks   (_check_dangerous_format)
        9.  Generic Cc/Cf control characters
        10. SQL injection patterns
        11. Code injection patterns
        12. Path traversal sequences
        13. Dangerous data URIs
        14. Markdown URL safety               (_check_markdown_urls)
        15. Pathological nesting              (_check_pathological_nesting)
        16. Abnormally long unbroken token
        17. Repeated character run

    Returns:
        NFC-normalised, validated text string.

    Raises:
        ValueError: on any violation.
    """
    if text is None:
        raise ValueError("OCR text must not be None")
    if not isinstance(text, str):
        raise ValueError(
            f"OCR text must be a string, got {type(text).__name__}"
        )
    if text.strip() == "":
        return ""
    if len(text) > max_length:
        raise ValueError(
            f"OCR text exceeds maximum length ({len(text):,} > {max_length:,} chars)"
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"OCR text contains invalid UTF-8 codepoints: {exc}")

    normalized = unicodedata.normalize("NFC", text)

    if "\x00" in normalized:
        raise ValueError("OCR text contains null bytes (U+0000)")

    _check_dangerous_format(normalized)

    _allowed_controls = {"\n", "\t", "\r"} | set(ZERO_WIDTH_CHARS)
    bad_chars = sorted({
        ch for ch in normalized
        if unicodedata.category(ch) in ("Cc", "Cf")
        and ch not in _allowed_controls
    })
    if bad_chars:
        raise ValueError(
            "Disallowed control or format character detected: "
            f"{[f'U+{ord(c):04X}' for c in bad_chars]}"
        )

    scan_text = re.sub(f"[{ZERO_WIDTH_CHARS}]", "", normalized)

    for pattern in SQL_INJECTION_PATTERNS:
        m = re.search(pattern, scan_text)
        if m:
            raise ValueError(
                f"SQL-injection-like pattern detected: '{m.group(0)}'"
            )

    for pattern in CODE_INJECTION_PATTERNS:
        m = re.search(pattern, scan_text)
        if m:
            raise ValueError(
                f"Code-injection-like pattern detected: '{m.group(0)}'"
            )

    if re.search(PATH_TRAVERSAL_PATTERN, normalized):
        raise ValueError("Path traversal sequence detected in OCR text")

    m = re.search(DANGEROUS_DATA_URI_PATTERN, normalized)
    if m:
        raise ValueError(f"Disallowed data URI detected: '{m.group(0)}'")

    _check_markdown_urls(scan_text)
    _check_pathological_nesting(normalized)

    if max((len(tok) for tok in normalized.split()), default=0) > 10_000:
        raise ValueError(
            "Abnormally long unbroken token detected — "
            "likely corrupted OCR output or embedded binary data"
        )

    if re.search(r"(.)\1{200,}", normalized):
        raise ValueError("Abnormal run of 200+ repeated characters detected")

    return normalized


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PAGE / DATABASE SAFETY VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _validate_page_number(page_number: int) -> int:
    """
    Validate a single page number before it is stored in PageContent.

    Page numbers are written to JSON and stored in the database as INTEGER.
    They must be a plain integer, >= 1, and <= _MAX_PDF_PAGES.

    Returns:
        The validated page number unchanged.

    Raises:
        TypeError:  if page_number is not a plain integer.
        ValueError: if page_number is out of range.
    """
    if page_number is None:
        raise ValueError("Page number must not be None")
    if isinstance(page_number, bool):
        raise TypeError("Page number must not be a boolean")
    if not isinstance(page_number, int):
        raise TypeError(
            f"Page number must be an integer, got {type(page_number).__name__}"
        )
    if page_number < _MIN_PAGE_NUMBER:
        raise ValueError(
            f"Page number must be >= {_MIN_PAGE_NUMBER}, got {page_number}"
        )
    if page_number > _MAX_PDF_PAGES:
        raise ValueError(
            f"Page number {page_number} exceeds the maximum allowed "
            f"({_MAX_PDF_PAGES})"
        )
    return page_number


def _validate_page_content(pc: PageContent) -> PageContent:
    """
    Validate a single PageContent object before it joins the output list.

    PageContent fields map directly to database columns:
        page_number  INTEGER NOT NULL
        text         TEXT NOT NULL

    Checks:
        1. pc is not None and is a PageContent instance
        2. page_number passes _validate_page_number
        3. text is not None and is a string
        4. text is not empty or whitespace-only
        5. text length <= MAX_TEXT_LENGTH (database TEXT column safety)
        6. text encodes to valid UTF-8

    Returns:
        The validated PageContent unchanged.

    Raises:
        TypeError:  if pc or its fields are the wrong type.
        ValueError: if any field fails validation.
    """
    if pc is None:
        raise ValueError("PageContent must not be None")
    if not isinstance(pc, PageContent):
        raise TypeError(
            f"Expected a PageContent instance, got {type(pc).__name__}"
        )

    _validate_page_number(pc.page_number)

    if pc.text is None:
        raise ValueError(f"Page {pc.page_number}: text field must not be None")
    if not isinstance(pc.text, str):
        raise TypeError(
            f"Page {pc.page_number}: text must be a string, "
            f"got {type(pc.text).__name__}"
        )
    if not pc.text.strip():
        raise ValueError(
            f"Page {pc.page_number}: text is empty or whitespace-only. "
            "Blank pages must be filtered out before reaching this point."
        )
    if len(pc.text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Page {pc.page_number}: text length {len(pc.text):,} chars "
            f"exceeds the database-safe limit of {MAX_TEXT_LENGTH:,} chars"
        )
    try:
        pc.text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"Page {pc.page_number}: text contains invalid UTF-8 and "
            f"cannot be stored safely: {exc}"
        )

    return pc


def _validate_pages_list(pages: list[PageContent]) -> list[PageContent]:
    """
    Validate the complete list of pages before serialisation to JSON
    and storage in the database.

    This is the final gate before data leaves the OCR pipeline.

    Checks:
        1.  Not None
        2.  Is a list
        3.  Not empty
        4.  Length <= _MAX_PDF_PAGES
        5.  Every element passes _validate_page_content
        6.  No duplicate page numbers
        7.  Page numbers are in strictly ascending order
        8.  Aggregate text size is within a safe bound

    Returns:
        The validated list unchanged.

    Raises:
        TypeError:  if pages is not a list or contains wrong-type elements.
        ValueError: on any structural or content violation.
    """
    if pages is None:
        raise ValueError("Pages list must not be None")
    if not isinstance(pages, list):
        raise TypeError(
            f"Pages must be a list, got {type(pages).__name__}"
        )
    if len(pages) == 0:
        raise ValueError(
            "Pages list is empty — no content was extracted from the document. "
            "Check that the PDF contains readable text."
        )
    if len(pages) > _MAX_PDF_PAGES:
        raise ValueError(
            f"Pages list has {len(pages)} entries, "
            f"exceeding the maximum of {_MAX_PDF_PAGES}."
        )

    for idx, pc in enumerate(pages):
        try:
            _validate_page_content(pc)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid page at list index {idx}: {exc}") from exc

    page_numbers = [pc.page_number for pc in pages]

    seen, duplicates = set(), set()
    for pn in page_numbers:
        (duplicates if pn in seen else seen).add(pn)
    if duplicates:
        raise ValueError(
            f"Duplicate page numbers in pages list: {sorted(duplicates)}. "
            "Each page number must appear exactly once."
        )

    if page_numbers != sorted(page_numbers):
        raise ValueError(
            f"Pages are not in ascending order. "
            f"Got page numbers: {page_numbers}. "
            "Pages must be ordered from first to last."
        )

    total_chars = sum(len(pc.text) for pc in pages)
    aggregate_limit = _MAX_PDF_PAGES * MAX_TEXT_LENGTH
    if total_chars > aggregate_limit:
        raise ValueError(
            f"Total text across all pages ({total_chars:,} chars) "
            f"exceeds the aggregate safety limit ({aggregate_limit:,} chars)."
        )

    return pages