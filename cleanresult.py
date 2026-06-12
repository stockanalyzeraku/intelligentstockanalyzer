from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

@dataclass
class TableType(Enum):
    QUALITATIVE = "qualitative"
    FINANCIAL   = "financial"
    
@dataclass
class CleanResult:
    page_number: int = 0
    clean_text: str = ""
    has_table: bool = False
    table_type: Optional[TableType] = None
    word_count: int = 0
    is_short: bool = False
    doc_type: str = "ANNUAL_REPORT"
    raw_tables: str = ""
    page_intent: dict[str, float] = field(default_factory=dict)
    company: str = ""
    year: int = 0
