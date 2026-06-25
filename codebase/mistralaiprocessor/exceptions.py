from typing import Any
from pathlib import Path
class FilePathError(Exception):
    def __init__(self, path: Path, label: str):
        self.path = path
        super().__init__(f"{path} : {label}")

        