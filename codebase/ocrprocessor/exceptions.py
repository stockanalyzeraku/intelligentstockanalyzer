from typing import Any
from pathlib import Path
class FilePathError(Exception):
    def __init__(self, path: Path, label: str):
        self.path = path
        super().__init__(f"{path} : {label}")

class FilenameValidationError(Exception):
    def __init__(self, filename: str, label: str):
        self.filename = filename
        super().__init__(f"{filename} : {label}")
        