

from pathlib import Path
import json
from codebase.cleaning.skelton import (
    MIN_YEAR,
    MAX_YEAR,
    ALLOWED_BASE,
    ALLOWED_EXTENSIONS
)
from codebase.cleaning.exceptions import (
    FilePathError
)
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

    return resolved
