from pathlib import Path
import unicodedata
import re
from urllib.parse import urlparse


from codebase.mistralaiprocessor.exceptions import (
    FilePathError
)
from codebase.mistralaiprocessor.skelton import (
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
    PRIVATE_IP_PATTERN
)

import json
from pathlib import Path



def _validate_filepath(path) -> Path:

    if path is None or str(path).strip() == "":
        raise FilePathError(path, "File Path is empty")

    resolved = Path(path).expanduser().resolve()

    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(
            f"{Path} : Path is not in base directory"
            f"(possible path traversal attempt)"
        )

    if not resolved.exists():
        raise FilePathError(f"{path} : File does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{path} : Path is not a regular file: {resolved}")

    # --- Structure check: uploads/script/year/file.ext ---
    try:
        relative_parts = resolved.relative_to(ALLOWED_BASE).parts
    except ValueError:
        raise ValueError(f"Could not compute relative path for: {resolved}")

    if len(relative_parts) != 3:
        raise ValueError(
            f"Expected structure 'uploads/<script>/<year>/<file>', "
            f"got {len(relative_parts)} segment(s) under uploads: {relative_parts}"
        )

    script_name, year_str, filename = relative_parts

    # --- Validate script segment ---
    if not script_name or not script_name.strip():
        raise ValueError("Script/folder name segment is empty")
    if script_name.startswith("."):
        raise ValueError(f"Script folder name looks hidden/invalid: '{script_name}'")

    # --- Validate year segment ---
    if not year_str.isdigit():
        raise ValueError(f"Year segment is not numeric: '{year_str}'")
    year = int(year_str)
    if not (MIN_YEAR <= year <= MAX_YEAR):
        raise ValueError(f"Year {year} is out of plausible range ({MIN_YEAR}-{MAX_YEAR})")

    # --- Validate filename / extension ---
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}', expected one of {ALLOWED_EXTENSIONS}"
        )
    if resolved.stat().st_size == 0:
        raise ValueError(f"File is empty: {resolved}")

    # --- Content-level validation by type ---
    if suffix == ".json":
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"File is not valid JSON: {e}")
        except UnicodeDecodeError:
            raise ValueError("JSON file is not valid UTF-8 text")

    elif suffix == ".pdf":
        with open(resolved, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            raise ValueError("File does not have a valid PDF header (missing '%PDF-')")

    return resolved




def _check_pathological_nesting(text: str) -> None:
    if re.search(rf"\[{{{NESTING_THRESHOLD},}}", text):
        raise ValueError(f"{NESTING_THRESHOLD}+ consecutive '[' characters (nesting bomb)")
    if re.search(rf"\({{{NESTING_THRESHOLD},}}", text):
        raise ValueError(f"{NESTING_THRESHOLD}+ consecutive '(' characters (nesting bomb)")
    if re.search(rf"^(>\s*){{{NESTING_THRESHOLD},}}", text, re.MULTILINE):
        raise ValueError(f"{NESTING_THRESHOLD}+ levels of blockquote nesting")


def _check_markdown_urls(text: str) -> None:
    for match in re.finditer(MARKDOWN_URL_PATTERN, text):
        url = match.group(1)
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if scheme in DANGEROUS_URI_SCHEMES:
            raise ValueError(f"Disallowed URI scheme '{scheme}:' in markdown link/image: {url}")
        if host in PRIVATE_HOSTS or re.match(PRIVATE_IP_PATTERN, host):
            raise ValueError(f"Markdown link/image targets a private/internal host (possible SSRF): {url}")


def _validate_ocr_text(text: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    if text is None:
        raise ValueError("Text must not be None")
    if not isinstance(text, str):
        raise ValueError(f"Text must be a string, got {type(text).__name__}")
    if text.strip() == "":
        raise ValueError("Text is empty or whitespace-only")
    if len(text) > max_length:
        raise ValueError(f"Text exceeds max allowed length ({len(text)} > {max_length})")

    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"Text contains invalid unicode for UTF-8 encoding: {e}")

    normalized = unicodedata.normalize("NFC", text)

    if "\x00" in normalized:
        raise ValueError("Text contains null bytes")

    # --- Bidi override characters: hard block ---
    found_bidi = [c for c in normalized if c in DANGEROUS_FORMAT_CHARS]
    if found_bidi:
        raise ValueError(
            f"Text contains explicit bidirectional override characters "
            f"(possible visual-spoofing attack): {[f'U+{ord(c):04X}' for c in found_bidi]}"
        )

    # --- Abnormal runs of invisible characters: block, but allow sparse use ---
    if re.search(f"[{ZERO_WIDTH_CHARS}]{{4,}}", normalized):
        raise ValueError("Abnormal run of invisible/zero-width characters detected")

    # --- Other control/format characters (excluding allowed whitespace & zero-width) ---
    allowed = {"\n", "\t", "\r"} | set(ZERO_WIDTH_CHARS)
    bad_chars = sorted({
        ch for ch in normalized
        if unicodedata.category(ch) in ("Cc", "Cf") and ch not in allowed
    })
    if bad_chars:
        raise ValueError(
            f"Text contains disallowed control characters: "
            f"{[f'U+{ord(c):04X}' for c in bad_chars]}"
        )

    # Strip zero-width chars ONLY for pattern-matching, so attackers can't
    # dodge detection by splitting keywords with invisible characters.
    scan_text = re.sub(f"[{ZERO_WIDTH_CHARS}]", "", normalized)

    for pattern in SQL_INJECTION_PATTERNS:
        match = re.search(pattern, scan_text)
        if match:
            raise ValueError(f"Text matched a SQL-injection-like pattern: '{match.group(0)}'")

    for pattern in CODE_INJECTION_PATTERNS:
        match = re.search(pattern, scan_text)
        if match:
            raise ValueError(f"Text matched a code-injection-like pattern: '{match.group(0)}'")

    if re.search(PATH_TRAVERSAL_PATTERN, normalized):
        raise ValueError("Text contains path traversal sequences ('../') in links/images")

    match = re.search(DANGEROUS_DATA_URI_PATTERN, normalized)
    if match:
        raise ValueError(f"Text contains a disallowed data URI: '{match.group(0)}'")

    _check_markdown_urls(normalized)
    _check_pathological_nesting(normalized)

    longest_token = max((len(tok) for tok in normalized.split()), default=0)
    if longest_token > 10_000:
        raise ValueError(
            f"Abnormally long unbroken token ({longest_token} chars) — "
            f"likely corrupted OCR or embedded binary data"
        )

    if re.search(r"(.)\1{200,}", normalized):
        raise ValueError("Abnormal run of 200+ repeated characters")

    return normalized