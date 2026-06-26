from pathlib import Path
import unicodedata
import re
from urllib.parse import urlparse
import ipaddress


from codebase.ocrprocessor.exceptions import (
    FilePathError
)
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
    MAX_REPLACEMENT_CHAR_DENSITY
)

import json

def _validate_filepath(path) -> Path:

    if path is None or str(path).strip() == "":
        raise FilePathError(path, "File Path is empty")

    resolved = Path(path).expanduser().resolve()

    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(
            path,
            f"{path} : Path is not in base directory (possible path traversal attempt)"
        )

    if not resolved.exists():
        raise FilePathError(path, f"{path} : File does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(path,f"{path} : Path is not a regular file: {resolved}")

    # --- Structure check: uploads/script/year/file.ext ---
    try:
        relative_parts = resolved.relative_to(ALLOWED_BASE).parts
    except ValueError:
        raise ValueError(f"Could not compute relative path for: {resolved}")

    if len(relative_parts) != 4:
        raise ValueError(
            f"Expected structure 'uploads/<script>/<year>/<file>', "
            f"got {len(relative_parts)} segment(s) under uploads: {relative_parts}"
        )

    script_name, year_str, file_type , filename = relative_parts

    # --- Validate script segment ---
    if not script_name or not script_name.strip() or not _SCRIPT_NAME_RE.match(script_name.upper()):
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
        return ""                               # blank pages are valid — caller skips them
    if len(text) > max_length:
        raise ValueError(f"Text exceeds max length ({len(text)} > {max_length})")

    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"Text contains invalid UTF-8 codepoints: {e}")

    normalized = unicodedata.normalize("NFC", text)

    if "\x00" in normalized:
        raise ValueError("Text contains null bytes")

    # ── Unicode format / spoofing / steganography attacks ────────────────────
    _check_dangerous_format(normalized)            # ← replaces 3 inline blocks

    # ── Generic Cc/Cf control character sweep ────────────────────────────────
    # Catches any remaining format characters not named above.
    # Note: U+2028/U+2029 (Zl/Zp) are already caught by _check_dangerous_format.
    _allowed_controls = {"\n", "\t", "\r"} | set(ZERO_WIDTH_CHARS)
    bad_chars = sorted({
        ch for ch in normalized
        if unicodedata.category(ch) in ("Cc", "Cf") and ch not in _allowed_controls
    })
    if bad_chars:
        raise ValueError(
            "Disallowed control/format character detected: "
            f"{[f'U+{ord(c):04X}' for c in bad_chars]}"
        )

    # ── Strip zero-width chars for pattern matching only ─────────────────────
    scan_text = re.sub(f"[{ZERO_WIDTH_CHARS}]", "", normalized)

    for pattern in SQL_INJECTION_PATTERNS:
        m = re.search(pattern, scan_text)
        if m:
            raise ValueError(f"SQL-injection-like pattern matched: '{m.group(0)}'")

    for pattern in CODE_INJECTION_PATTERNS:
        m = re.search(pattern, scan_text)
        if m:
            raise ValueError(f"Code-injection-like pattern matched: '{m.group(0)}'")

    if re.search(PATH_TRAVERSAL_PATTERN, normalized):
        raise ValueError("Path traversal sequence detected in text")

    m = re.search(DANGEROUS_DATA_URI_PATTERN, normalized)
    if m:
        raise ValueError(f"Disallowed data URI detected: '{m.group(0)}'")

    _check_markdown_urls(scan_text)
    _check_pathological_nesting(normalized)

    if max(len(tok) for tok in normalized.split() or [""]) > 10_000:
        raise ValueError("Abnormally long unbroken token — likely corrupted OCR or binary data")

    if re.search(r"(.)\1{200,}", normalized):
        raise ValueError("Abnormal run of 200+ repeated characters")

    return normalized

def _validate_output_path(output_path: str) -> Path:
    """Validate a write-destination path. The file need not exist yet."""
    if not output_path or str(output_path).strip() == "":
        raise FilePathError(output_path, "Output path is empty")

    resolved = Path(output_path).expanduser().resolve()

    try:
        resolved.relative_to(ALLOWED_BASE)
    except ValueError:
        raise FilePathError(resolved, "Output path is outside allowed base directory")

    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise FilePathError(resolved, f"Output extension '{resolved.suffix}' not allowed")

    # Validate parent directory exists or can be created
    if not resolved.parent.exists():
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FilePathError(resolved, f"Cannot create output directory: {e}")

    return resolved

def _check_dangerous_format(text: str) -> None:
    """Detect Unicode-based visual spoofing, steganography and injection attacks.

    This function must be called on NFC-normalised text BEFORE zero-width
    stripping, so that both the raw and stripped forms are inspectable.

    Attack vectors covered:
        - Trojan Source (CVE-2021-42574): Bidi overrides visually reverse
          text to hide malicious content from human reviewers.
        - Directional mark injection: invisible marks alter render direction.
        - JS string injection: U+2028/U+2029 are category Zl/Zp, not Cc/Cf,
          so they bypass generic control-character sweeps.
        - Tag block steganography: U+E0000–U+E007F are invisible and have
          had no legitimate use since Unicode 6.0 (2010).
        - Interlinear annotation abuse: anchors with no prose use.
        - Object replacement flooding: masks binary data lengths.
        - Zero-width keyword splitting: evasion of injection detectors.

    Args:
        text: NFC-normalised string, exactly as received from OCR output.

    Raises:
        ValueError: Descriptive error naming the specific attack vector
                    and the exact offending codepoints (e.g. U+202E).
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
    # IMPORTANT: These are Unicode category Zl/Zp — they are NOT caught by
    # the Cc/Cf control-character sweep below. This check must be explicit.
    found_sep = [c for c in text if c in SEPARATOR_INJECTION_CHARS]
    if found_sep:
        raise ValueError(
            "Line/paragraph separator character detected "
            "(U+2028/U+2029 can escape JavaScript string literals and "
            "cause silent line breaks inside JSON values). "
            f"Offending codepoints: {[f'U+{ord(c):04X}' for c in found_sep]}"
        )

    # ── 4. Unicode Tag block (U+E0000–U+E007F) — invisible steganography ──────
    lo, hi = TAG_BLOCK_RANGE
    tag_chars = [c for c in text if lo <= ord(c) <= hi]
    if tag_chars:
        raise ValueError(
            "Unicode Tag block character detected "
            "(invisible steganography — deprecated since Unicode 6.0 with "
            "no legitimate use in any modern document). "
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
    # U+FFFC is category So — NOT caught by the Cc/Cf sweep. A small number
    # may appear in corrupted OCR (legitimate). A high density signals that
    # binary content has been embedded and masked.
    obj_count = text.count(OBJECT_REPLACEMENT_CHAR)
    if obj_count > MAX_REPLACEMENT_CHAR_DENSITY:
        raise ValueError(
            f"Excessive object replacement characters detected: {obj_count} "
            f"(threshold: {MAX_REPLACEMENT_CHAR_DENSITY}). "
            "Possible binary data embedded and masked inside OCR output."
        )

    # ── 7. Zero-width character run — keyword-splitting evasion ──────────────
    # Individual zero-width chars are allowed (legitimate in Arabic/Indic
    # scripts). Runs of ≥4 consecutive chars are never legitimate and are a
    # known technique to split injection keywords across invisible boundaries.
    if re.search(f"[{ZERO_WIDTH_CHARS}]{{4,}}", text):
        raise ValueError(
            "Abnormal run of ≥4 consecutive zero-width characters detected. "
            "This pattern is used to split injection keywords (e.g. 's\u200be\u200bl\u200be\u200bct') "
            "to evade pattern-based security scanners."
        )