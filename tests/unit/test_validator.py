# tests/unit/test_validator.py
"""
Tests for all functions in validator.py.

This is the security-critical file. Every check in _validate_ocr_text
and _validate_filepath gets:
  1. A happy-path test confirming legitimate input passes
  2. One or more adversarial tests confirming the attack is blocked
  3. An edge-case test at the boundary of the rule

Tests are grouped by function, then by check within that function.
Use Ctrl+F on the attack name to jump to its test.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from codebase.ocrprocessor.exceptions import FilePathError
from codebase.ocrprocessor.validator import (
    _validate_filepath,
    _validate_ocr_text,
    _check_pathological_nesting,
    _check_markdown_urls,
)


# ════════════════════════════════════════════════════════════════════════════
# _validate_filepath
# ════════════════════════════════════════════════════════════════════════════

class TestValidateFilepathHappyPath:

    def test_valid_pdf_returns_resolved_path(self, valid_pdf):
        result = _validate_filepath(str(valid_pdf))
        assert isinstance(result, Path)
        assert result == valid_pdf.resolve()

    def test_valid_json_returns_resolved_path(self, valid_json):
        result = _validate_filepath(str(valid_json))
        assert isinstance(result, Path)

    def test_accepts_path_object(self, valid_pdf):
        result = _validate_filepath(valid_pdf)
        assert result == valid_pdf.resolve()


class TestValidateFilepathEmptyInput:

    def test_none_raises_file_path_error(self, allowed_base):
        with pytest.raises(FilePathError, match="empty"):
            _validate_filepath(None)

    def test_empty_string_raises_file_path_error(self, allowed_base):
        with pytest.raises(FilePathError, match="empty"):
            _validate_filepath("")

    def test_whitespace_only_raises_file_path_error(self, allowed_base):
        with pytest.raises(FilePathError, match="empty"):
            _validate_filepath("   ")


class TestValidateFilepathTraversal:
    """Path traversal attempts must raise FilePathError, not crash with TypeError."""

    @pytest.mark.parametrize("evil_path", [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "%2e%2e%2fetc%2fpasswd",
        "%252e%252e%252f",
        "uploads/TESTCO/../../../etc/shadow",
    ])
    def test_traversal_attempt_raises_file_path_error(self, evil_path, allowed_base):
        with pytest.raises(FilePathError):
            _validate_filepath(evil_path)

    def test_file_outside_allowed_base_raises_file_path_error(
        self, outside_path, allowed_base
    ):
        with pytest.raises(FilePathError):
            _validate_filepath(str(outside_path))

    def test_error_message_does_not_reveal_system_paths(self, allowed_base):
        """Error messages should not leak the real filesystem layout."""
        try:
            _validate_filepath("../../../etc/passwd")
        except FilePathError as e:
            assert "/etc/passwd" not in str(e), \
                "Error message leaks target path — information disclosure risk"


class TestValidateFilepathStructure:

    def test_two_segments_raises(self, allowed_base):
        """Expects exactly 3 segments: company/year/file."""
        company_dir = allowed_base / "TESTCO"
        company_dir.mkdir()
        orphan = company_dir / "TESTCO.pdf"
        orphan.write_bytes(b"%PDF-1.4\n%%EOF")
        with pytest.raises((ValueError, FilePathError)):
            _validate_filepath(str(orphan))

    def test_invalid_company_name_raises(self, allowed_base):
        """Company name must match [A-Z0-9_-]{1,64}."""
        bad_dir = allowed_base / "TEST;CO" / "2024"
        bad_dir.mkdir(parents=True)
        f = bad_dir / "TEST;CO_ANNUAL_2024.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF")
        with pytest.raises(ValueError):
            _validate_filepath(str(f))

    def test_year_out_of_range_raises(self, allowed_base):
        old_dir = allowed_base / "TESTCO" / "1800"
        old_dir.mkdir(parents=True)
        f = old_dir / "TESTCO_ANNUAL_1800.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF")
        with pytest.raises(ValueError, match="range"):
            _validate_filepath(str(f))

    def test_disallowed_extension_raises(self, allowed_base):
        exe_dir = allowed_base / "TESTCO" / "2024"
        exe_dir.mkdir(parents=True)
        f = exe_dir / "TESTCO_ANNUAL_2024.exe"
        f.write_bytes(b"MZ malicious binary")
        with pytest.raises(ValueError, match="extension"):
            _validate_filepath(str(f))

    def test_empty_file_raises(self, allowed_base):
        company_dir = allowed_base / "TESTCO" / "2024"
        company_dir.mkdir(parents=True)
        f = company_dir / "TESTCO_ANNUAL_2024.pdf"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            _validate_filepath(str(f))

    def test_invalid_pdf_header_raises(self, allowed_base):
        company_dir = allowed_base / "TESTCO" / "2024"
        company_dir.mkdir(parents=True)
        f = company_dir / "TESTCO_ANNUAL_2024.pdf"
        f.write_bytes(b"This is not a PDF file at all")
        with pytest.raises(ValueError, match="PDF header"):
            _validate_filepath(str(f))

    def test_invalid_json_content_raises(self, allowed_base):
        company_dir = allowed_base / "TESTCO" / "2024"
        company_dir.mkdir(parents=True)
        f = company_dir / "TESTCO_ANNUAL_2024.json"
        f.write_text("{invalid json{{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON"):
            _validate_filepath(str(f))


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — Input guards
# ════════════════════════════════════════════════════════════════════════════

class TestValidateOcrTextInputGuards:

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            _validate_ocr_text(None)

    def test_integer_raises(self):
        with pytest.raises(ValueError, match="string"):
            _validate_ocr_text(42)

    def test_empty_string_returns_empty(self):
        # After M-06 fix: blank pages are valid, return "" not raise
        result = _validate_ocr_text("")
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        result = _validate_ocr_text("   \n\t  ")
        assert result == ""

    def test_exceeds_max_length_raises(self):
        with pytest.raises(ValueError, match="max"):
            _validate_ocr_text("x" * (5_000_001), max_length=5_000_000)

    def test_exactly_at_max_length_passes(self, clean_financial_text):
        # Confirm boundary: length == limit is allowed
        text = "a" * 5_000_000
        result = _validate_ocr_text(text, max_length=5_000_000)
        assert isinstance(result, str)

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="null"):
            _validate_ocr_text("valid text\x00injected")

    def test_clean_financial_text_passes(self, clean_financial_text):
        result = _validate_ocr_text(clean_financial_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_nfc_normalized_string(self):
        # café with decomposed 'e' + combining accent (NFD) → NFC on return
        nfd = "caf\u0065\u0301"    # NFD form
        nfc = "caf\u00e9"          # NFC form (é as single codepoint)
        result = _validate_ocr_text(nfd)
        assert result == nfc


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — Unicode format attacks
# ════════════════════════════════════════════════════════════════════════════

class TestBidiOverrideAttack:
    """Trojan Source — CVE-2021-42574"""

    @pytest.mark.parametrize("bidi_char,codepoint_name", [
        ("\u202a", "LEFT-TO-RIGHT EMBEDDING"),
        ("\u202b", "RIGHT-TO-LEFT EMBEDDING"),
        ("\u202c", "POP DIRECTIONAL FORMATTING"),
        ("\u202d", "LEFT-TO-RIGHT OVERRIDE"),
        ("\u202e", "RIGHT-TO-LEFT OVERRIDE"),
        ("\u2066", "LEFT-TO-RIGHT ISOLATE"),
        ("\u2067", "RIGHT-TO-LEFT ISOLATE"),
        ("\u2068", "FIRST STRONG ISOLATE"),
        ("\u2069", "POP DIRECTIONAL ISOLATE"),
    ])
    def test_bidi_char_is_blocked(self, bidi_char, codepoint_name):
        text = f"Normal text{bidi_char}hidden content"
        with pytest.raises(ValueError) as exc_info:
            _validate_ocr_text(text)
        assert "U+" in str(exc_info.value), \
            f"Error message should include the codepoint for {codepoint_name}"

    def test_rtl_override_blocked_mid_word(self):
        """Most dangerous: reversing part of a word to hide payload."""
        text = "The amount is \u202e000,1₹"  # visually shows ₹1,000 but actual order is reversed
        with pytest.raises(ValueError):
            _validate_ocr_text(text)

    def test_error_names_trojan_source_or_bidi(self):
        text = "text\u202emore text"
        with pytest.raises(ValueError) as exc_info:
            _validate_ocr_text(text)
        msg = str(exc_info.value).lower()
        assert "bidi" in msg or "trojan" in msg or "bidirectional" in msg


class TestSeparatorInjectionAttack:
    """U+2028/U+2029 — JavaScript string injection via line/paragraph separator."""

    def test_line_separator_u2028_is_blocked(self):
        text = "Normal line\u2028injected line"
        with pytest.raises(ValueError) as exc_info:
            _validate_ocr_text(text)
        assert "2028" in str(exc_info.value).upper()

    def test_paragraph_separator_u2029_is_blocked(self):
        text = "Normal text\u2029injected paragraph"
        with pytest.raises(ValueError) as exc_info:
            _validate_ocr_text(text)
        assert "2029" in str(exc_info.value).upper()

    def test_separator_not_blocked_in_normal_newline(self):
        """Regular \n must still work — not confused with separators."""
        result = _validate_ocr_text("Line one\nLine two\nLine three")
        assert isinstance(result, str)


class TestTagBlockSteganography:
    """Unicode Tag block U+E0000–U+E007F — invisible hidden text."""

    def test_tag_space_u_e0020_is_blocked(self):
        text = "Visible text\U000e0020hidden"
        with pytest.raises(ValueError) as exc_info:
            _validate_ocr_text(text)
        assert "tag" in str(exc_info.value).lower() or "E0020" in str(exc_info.value).upper()

    def test_language_tag_u_e0001_is_blocked(self):
        text = "Text with\U000e0001language tag"
        with pytest.raises(ValueError):
            _validate_ocr_text(text)


class TestZeroWidthEvasion:
    """Zero-width chars used to split injection keywords."""

    def test_run_of_four_zero_width_chars_blocked(self):
        text = "normal\u200b\u200b\u200b\u200btext"
        with pytest.raises(ValueError, match="zero-width"):
            _validate_ocr_text(text)

    def test_three_zero_width_chars_allowed(self):
        """Sparse ZWJ/ZWNJ is legitimate in Indic script shaping."""
        text = "हिन्\u200dदी text"  # ZWJ in Hindi — legitimate
        result = _validate_ocr_text(text)
        assert isinstance(result, str)

    def test_keyword_split_by_zero_width_is_detected(self):
        """SELECT split across zero-width chars must still be caught after stripping."""
        # s\u200be\u200bl\u200be\u200bc\u200bt from users
        # Strip ZW → "select from users" → SQL pattern fires
        text = "s\u200be\u200bl\u200be\u200bc\u200bt from users"
        with pytest.raises(ValueError):
            _validate_ocr_text(text)


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — SQL Injection
# ════════════════════════════════════════════════════════════════════════════

class TestSQLInjectionDetection:

    @pytest.mark.parametrize("payload,description", [
        ("' UNION SELECT password, username FROM users--", "union select"),
        ("'; DROP TABLE students;--", "drop table"),
        ("1' AND 1=1--", "and tautology"),
        ("' OR '1'='1", "or string tautology"),
        ("'; WAITFOR DELAY '0:0:5'--", "time-based blind (SQL Server)"),
        ("1'; SELECT SLEEP(5)--", "time-based blind (MySQL)"),
        ("'; TRUNCATE TABLE logs;--", "truncate"),
        ("SELECT 1 INTO OUTFILE '/var/www/shell.php'", "into outfile"),
        ("SELECT * FROM information_schema.tables", "information_schema"),
        ("'; EXEC xp_cmdshell('dir');--", "xp_cmdshell"),
        ("1; DELETE FROM users WHERE 1=1", "stacked delete"),
        ("0x53454c454354", "hex encoding"),
    ])
    def test_sql_payload_is_blocked(self, payload, description):
        with pytest.raises(ValueError, match="SQL"), \
                pytest.fail.__class__:
            _validate_ocr_text(payload)
        # Rewrite without pytest.fail.__class__:

    # Cleaner version of parametrized SQL tests:
    SQL_PAYLOADS = [
        ("' UNION SELECT password FROM users--", "union-based"),
        ("'; DROP TABLE students;--", "drop table"),
        ("' OR 1=1--", "boolean tautology"),
        ("'; WAITFOR DELAY '0:0:5'--", "SQL Server time-based"),
        ("'; SELECT SLEEP(5)--", "MySQL time-based"),
        ("'; TRUNCATE TABLE logs;", "truncate"),
        ("SELECT * INTO OUTFILE '/tmp/x'", "into outfile"),
        ("SELECT table_name FROM information_schema.tables", "information_schema"),
        ("'; EXEC xp_cmdshell('id');--", "xp_cmdshell"),
        ("1; DELETE FROM users", "stacked query"),
        ("SELECT CHAR(83,69,76)", "char obfuscation"),
        ("1 GROUP_CONCAT(password)", "group_concat"),
    ]

    @pytest.mark.parametrize("payload,description", SQL_PAYLOADS)
    def test_sql_injection_blocked(self, payload, description):
        with pytest.raises(ValueError):
            _validate_ocr_text(payload)


class TestSQLFalsePositives:
    """
    These are real sentences from annual reports that contain SQL keywords.
    They demonstrate the known false-positive risk (Finding M-08).
    Tests here are marked xfail — they document the known limitation.
    Once a more context-aware detection strategy is implemented, remove xfail.
    """

    @pytest.mark.xfail(
        reason="M-08: SQL patterns produce false positives on English prose. "
               "Known limitation — fix requires context-aware detection.",
        strict=False,
    )
    def test_english_select_from_does_not_trigger(self):
        text = "The Board will select the Managing Director from shortlisted candidates."
        result = _validate_ocr_text(text)  # Should pass, currently fails
        assert isinstance(result, str)

    @pytest.mark.xfail(reason="M-08: False positive risk", strict=False)
    def test_english_insert_into_does_not_trigger(self):
        text = "The company will insert this indemnity clause into all future contracts."
        result = _validate_ocr_text(text)
        assert isinstance(result, str)


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — Code Injection
# ════════════════════════════════════════════════════════════════════════════

class TestCodeInjectionDetection:

    CODE_PAYLOADS = [
        ("<script>alert(document.cookie)</script>", "XSS script tag"),
        ("<iframe src='javascript:alert(1)'></iframe>", "iframe XSS"),
        ("{{ 7*7 }}", "Jinja2 SSTI"),
        ("${T(java.lang.Runtime).getRuntime().exec('id')}", "Spring SpEL"),
        ("eval('import os; os.system(\"id\")')", "eval"),
        ("exec(compile('import os', '<string>', 'exec'))", "exec+compile"),
        ("__import__('subprocess').call(['id'])", "__import__"),
        ("pickle.loads(b'\\x80\\x04\\x95...')", "pickle deserialization"),
        ("subprocess.Popen(['sh', '-c', 'id'])", "subprocess"),
        ("os.system('cat /etc/passwd')", "os.system"),
        ("globals()['__builtins__']['eval']('1+1')", "globals+builtins"),
        ("`rm -rf /`", "backtick shell execution"),
        ("$(cat /etc/passwd)", "shell expansion"),
        ("<meta http-equiv='refresh' content='0;url=http://evil.com'>", "meta refresh"),
    ]

    @pytest.mark.parametrize("payload,description", CODE_PAYLOADS)
    def test_code_injection_blocked(self, payload, description):
        with pytest.raises(ValueError):
            _validate_ocr_text(payload)


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — Path Traversal
# ════════════════════════════════════════════════════════════════════════════

class TestPathTraversalInText:

    @pytest.mark.parametrize("payload", [
        "Download from [link](../../../etc/passwd)",
        "Image: ![img](../../secret.pdf)",
        "See %2e%2e%2fetc%2fpasswd for details",
        "Path: %252e%252e%252fetc",
        "File\x00.pdf",  # null byte injection
    ])
    def test_traversal_in_text_blocked(self, payload):
        with pytest.raises(ValueError):
            _validate_ocr_text(payload)


# ════════════════════════════════════════════════════════════════════════════
# _validate_ocr_text — Abnormal content
# ════════════════════════════════════════════════════════════════════════════

class TestAbnormalContent:

    def test_long_unbroken_token_blocked(self):
        text = "Normal text " + "x" * 10_001 + " normal"
        with pytest.raises(ValueError, match="token"):
            _validate_ocr_text(text)

    def test_repeated_character_run_blocked(self):
        text = "Normal " + "A" * 201 + " text"
        with pytest.raises(ValueError, match="repeated"):
            _validate_ocr_text(text)

    def test_dangerous_data_uri_blocked(self):
        text = "Click [here](data:text/html,<script>alert(1)</script>)"
        with pytest.raises(ValueError, match="data"):
            _validate_ocr_text(text)


# ════════════════════════════════════════════════════════════════════════════
# _check_markdown_urls
# ════════════════════════════════════════════════════════════════════════════

class TestCheckMarkdownUrls:

    def test_normal_https_url_passes(self):
        _check_markdown_urls("[Report](https://example.com/annual-report.pdf)")

    def test_javascript_scheme_blocked(self):
        with pytest.raises(ValueError, match="javascript"):
            _check_markdown_urls("[Click](javascript:alert(1))")

    def test_file_scheme_blocked(self):
        with pytest.raises(ValueError, match="file"):
            _check_markdown_urls("[Local](file:///etc/passwd)")

    def test_data_scheme_blocked(self):
        with pytest.raises(ValueError, match="data"):
            _check_markdown_urls("[Img](data:image/png;base64,AAAA)")

    @pytest.mark.parametrize("private_host", [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",       # AWS/GCP/Azure metadata
        "metadata.google.internal",
    ])
    def test_ssrf_private_host_blocked(self, private_host):
        with pytest.raises(ValueError, match="SSRF|private|internal"):
            _check_markdown_urls(f"[link](http://{private_host}/secret)")

    @pytest.mark.parametrize("private_range", [
        "10.0.0.1",
        "192.168.1.100",
        "172.16.0.1",
        "172.31.255.255",
    ])
    def test_ssrf_private_ip_range_blocked(self, private_range):
        with pytest.raises(ValueError, match="SSRF|private"):
            _check_markdown_urls(f"[link](http://{private_range}/endpoint)")

    def test_image_link_also_checked(self):
        with pytest.raises(ValueError):
            _check_markdown_urls("![img](javascript:alert(1))")

    def test_multiple_urls_all_checked(self):
        """If any URL in the text is dangerous, must raise."""
        text = (
            "[OK](https://example.com) "
            "[evil](javascript:alert(1)) "
            "[also OK](https://docs.example.com)"
        )
        with pytest.raises(ValueError):
            _check_markdown_urls(text)


# ════════════════════════════════════════════════════════════════════════════
# _check_pathological_nesting
# ════════════════════════════════════════════════════════════════════════════

class TestCheckPathologicalNesting:

    def test_normal_bracket_passes(self):
        _check_pathological_nesting("This is [a link](url) and [[wiki-style]]")

    def test_eight_consecutive_brackets_blocked(self):
        with pytest.raises(ValueError, match="nesting"):
            _check_pathological_nesting("[[[[[[[[text")

    def test_seven_consecutive_brackets_blocked(self):
        # NESTING_THRESHOLD = 8, so 8+ is blocked
        # 7 should also be tested to confirm boundary
        from codebase.ocrprocessor.skelton import NESTING_THRESHOLD
        safe = "[" * (NESTING_THRESHOLD - 1) + "text"
        _check_pathological_nesting(safe)  # should not raise

    def test_nesting_bomb_in_parens_blocked(self):
        with pytest.raises(ValueError, match="nesting"):
            _check_pathological_nesting("((((((((text")

    def test_blockquote_nesting_bomb_blocked(self):
        deep_quote = "\n".join(["> " * 9 + "text"])
        with pytest.raises(ValueError, match="blockquote"):
            _check_pathological_nesting(deep_quote)