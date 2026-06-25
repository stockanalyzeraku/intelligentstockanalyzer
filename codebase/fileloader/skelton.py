from dataclasses import dataclass, field
from datetime import datetime
import re
#patterns
FILENAME_PATTERN = re.compile(
    r"^(?P<scrip>[A-Za-z0-9]+)_(?P<year>\d{4})_(?P<filetype>pdf)\.pdf$")

#classes
@dataclass
class UploadResult:
    filename: str
    status: str            # "SUCCESS" or "FAILED"
    reason: str = ""        # empty on success, error message on failure
    scrip: str | None = None
    year: str | None = None
    destination_path: str | None = None
    file_type:str = ""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
