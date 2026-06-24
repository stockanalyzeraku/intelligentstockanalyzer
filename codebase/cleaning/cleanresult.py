"""
Classes
-------
TableType   : Enum distinguishing financial vs qualitative tables.
CleanResult : Dataclass holding all per-page artefacts produced by TextCleaner.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from codebase.cleaning.struct import CleanResult


class TableType(Enum):
    """Classification of a markdown table found on a page."""

    QUALITATIVE = "qualitative"
    FINANCIAL   = "financial"



# @dataclass
# class CleanResult:
#     """
#     Container for all artefacts produced while cleaning a single PDF page.
#     """

#     page_number:  int                    = 0
#     clean_text:   str                    = ""
#     has_table:    bool                   = False
#     table_type:   Optional[TableType]    = None
#     word_count:   int                    = 0
#     is_short:     bool                   = False
#     doc_type:     str                    = "ANNUAL_REPORT"
#     raw_tables:   str                    = ""
#     page_intent:  list                   = field(default_factory=list)
#     company:      str                    = ""
#     year:         int                    = 0