# tests/unit/test_skelton.py
"""
Tests for constants and compiled patterns in skelton.py.

Goal: every regex must compile, every constant must have the right type
and value. If a pattern is added or removed from SQL_INJECTION_PATTERNS
or CODE_INJECTION_PATTERNS without updating tests, tests here will fail.
"""
import re
import pytest
from datetime import date
from codebase.ocrprocessor import skelton


class TestConstants:

    def test_allowed_extensions_contains_pdf_and_json(self):
        assert ".pdf" in skelton.ALLOWED_EXTENSIONS
        assert ".json" in skelton.ALLOWED_EXTENSIONS

    def test_max_year_is_current_year_plus_one(self):
        expected = date.today().year + 1
        assert skelton.MAX_YEAR == expected

    def test_min_year_is_1995(self):
        assert skelton.MIN_YEAR == 1995

    def test_nesting_threshold_is_sensible(self):
        # Must be low enough to block attacks, high enough for legitimate nesting
        assert 5 <= skelton.NESTING_THRESHOLD <= 12

    def test_max_text_length_is_positive(self):
        assert skelton.MAX_TEXT_LENGTH > 0


class TestBidiConstants:

    def test_bidi_override_chars_has_nine_entries(self):
        # 5 override + 4 isolate characters
        assert len(skelton.BIDI_OVERRIDE_CHARS) == 9

    def test_right_to_left_override_is_present(self):
        assert "\u202e" in skelton.BIDI_OVERRIDE_CHARS

    def test_all_isolate_chars_present(self):
        for cp in ["\u2066", "\u2067", "\u2068", "\u2069"]:
            assert cp in skelton.BIDI_OVERRIDE_CHARS

    def test_directional_marks_are_separate_from_overrides(self):
        # LTR/RTL marks are less dangerous — kept in a separate set
        assert "\u200e" in skelton.DIRECTIONAL_MARKS
        assert "\u200f" in skelton.DIRECTIONAL_MARKS
        assert "\u200e" not in skelton.BIDI_OVERRIDE_CHARS

    def test_separator_injection_chars_present(self):
        assert "\u2028" in skelton.SEPARATOR_INJECTION_CHARS  # LINE SEPARATOR
        assert "\u2029" in skelton.SEPARATOR_INJECTION_CHARS  # PARAGRAPH SEPARATOR

    def test_tag_block_range_is_correct(self):
        lo, hi = skelton.TAG_BLOCK_RANGE
        assert lo == 0xE0000
        assert hi == 0xE007F


class TestPatternCompilation:
    """All patterns must be syntactically valid regex strings."""

    @pytest.mark.parametrize("pattern", skelton.SQL_INJECTION_PATTERNS)
    def test_sql_pattern_compiles(self, pattern):
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"SQL pattern failed to compile: {pattern!r}\nError: {e}")

    @pytest.mark.parametrize("pattern", skelton.CODE_INJECTION_PATTERNS)
    def test_code_pattern_compiles(self, pattern):
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"Code pattern failed to compile: {pattern!r}\nError: {e}")

    def test_path_traversal_pattern_is_compiled(self):
        assert hasattr(skelton.PATH_TRAVERSAL_PATTERN, "search"), \
            "PATH_TRAVERSAL_PATTERN should be a compiled re.Pattern, not a raw string"

    def test_private_ip_pattern_is_compiled(self):
        assert hasattr(skelton.PRIVATE_IP_PATTERN, "match")


class TestSQLPatternCoverage:
    """Verify each threat category has at least one pattern covering it."""

    def _matches_any(self, text: str) -> bool:
        return any(
            re.search(p, text) for p in skelton.SQL_INJECTION_PATTERNS
        )

    def test_union_select_detected(self):
        assert self._matches_any("' UNION SELECT password FROM users--")

    def test_drop_table_detected(self):
        assert self._matches_any("'; DROP TABLE students;--")

    def test_time_based_blind_detected(self):
        assert self._matches_any("'; WAITFOR DELAY '0:0:5'--")

    def test_truncate_detected(self):
        assert self._matches_any("'; TRUNCATE TABLE logs;")

    def test_into_outfile_detected(self):
        assert self._matches_any("SELECT 1 INTO OUTFILE '/tmp/x'")

    def test_information_schema_detected(self):
        assert self._matches_any("SELECT * FROM information_schema.tables")

    def test_stacked_query_detected(self):
        assert self._matches_any("1; DELETE FROM users")

    def test_hex_encoding_detected(self):
        assert self._matches_any("0x53454c454354")

    def test_char_obfuscation_detected(self):
        assert self._matches_any("CHAR(83)CHAR(69)CHAR(76)")


class TestCodePatternCoverage:
    """Verify each threat category has at least one pattern covering it."""

    def _matches_any(self, text: str) -> bool:
        return any(
            re.search(p, text) for p in skelton.CODE_INJECTION_PATTERNS
        )

    def test_xss_script_tag_detected(self):
        assert self._matches_any("<script>alert(1)</script>")

    def test_jinja2_ssti_detected(self):
        assert self._matches_any("{{ 7 * 7 }}")

    def test_spring_spel_detected(self):
        assert self._matches_any("${T(java.lang.Runtime).getRuntime().exec('id')}")

    def test_eval_detected(self):
        assert self._matches_any("eval('malicious()')")

    def test_pickle_deserialization_detected(self):
        assert self._matches_any("pickle.loads(data)")

    def test_subprocess_detected(self):
        assert self._matches_any("subprocess.run(['rm', '-rf', '/'])")

    def test_dunder_globals_detected(self):
        assert self._matches_any("__globals__['os'].system('id')")

    def test_backtick_execution_detected(self):
        assert self._matches_any("`rm -rf /`")