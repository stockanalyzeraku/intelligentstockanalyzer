from dataclasses import dataclass, field
from datetime import datetime

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
