# tests/unit/test_exceptions.py
"""
Tests for FilePathError.

Small file — but important because FilePathError is the primary
exception used by the security-critical _validate_filepath function.
If the constructor is wrong, the entire exception chain silently breaks.
"""
import pytest
from pathlib import Path
from codebase.ocrprocessor.exceptions import FilePathError


class TestFilePathError:

    def test_inherits_from_exception(self):
        err = FilePathError(Path("/some/path"), "something went wrong")
        assert isinstance(err, Exception)

    def test_str_contains_path_and_label(self):
        err = FilePathError(Path("/uploads/file.pdf"), "File does not exist")
        msg = str(err)
        assert "/uploads/file.pdf" in msg
        assert "File does not exist" in msg

    def test_path_attribute_is_set(self):
        p = Path("/uploads/secret.pdf")
        err = FilePathError(p, "label")
        assert err.path == p

    def test_can_be_raised_and_caught(self):
        with pytest.raises(FilePathError) as exc_info:
            raise FilePathError(Path("/x"), "test label")
        assert "test label" in str(exc_info.value)

    def test_string_path_accepted(self):
        # path parameter typed as Path but often passed as str in practice
        err = FilePathError("/str/path", "label")
        assert "/str/path" in str(err)