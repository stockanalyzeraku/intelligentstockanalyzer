"""
cleaning/cleanresult.py
=======================
Shared data-container classes used across the entire cleaning pipeline.

Classes
-------
TableType   : Enum distinguishing financial vs qualitative tables.
CleanResult : Dataclass holding all per-page artefacts produced by TextCleaner.

Notes
-----
- ``TableType`` is a plain ``Enum`` — it must NOT be decorated with
  ``@dataclass`` (enums manage their own ``__init__``).
- ``CleanResult`` uses ``field(default_factory=list)`` for the
  ``page_intent`` list so every instance gets its own list object,
  preventing the classic shared-mutable-default bug.
- ``raw_tables`` is a ``str`` because ``TableExtractor.strip_tables``
  returns a newline-joined string of pipe-table blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# TableType
# ---------------------------------------------------------------------------

class TableType(Enum):
    """Classification of a markdown table found on a page."""

    QUALITATIVE = "qualitative"
    FINANCIAL   = "financial"


# ---------------------------------------------------------------------------
# CleanResult
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    """
    Container for all artefacts produced while cleaning a single PDF page.

    Attributes
    ----------
    page_number : int
        1-based page index from the source PDF.
    clean_text : str
        Narrative prose after removing OCR noise, headers/footers, ToC
        lines, and pipe-tables.
    has_table : bool
        True when at least one valid markdown pipe-table was detected on
        the page before table extraction.
    table_type : TableType | None
        ``TableType.FINANCIAL`` or ``TableType.QUALITATIVE`` when
        ``has_table`` is True; ``None`` otherwise.
    word_count : int
        Word count of ``clean_text`` *after* tables have been stripped.
    is_short : bool
        True when ``word_count`` falls below the configured minimum
        threshold (default 20 words).  Short pages are likely dividers,
        cover pages, or OCR noise and are skipped during embedding.
    doc_type : str
        Document category, e.g. ``"ANNUAL_REPORT"``, ``"EARNINGS_CALL"``.
    raw_tables : str
        Pipe-table blocks extracted verbatim from the page, joined by
        ``"\\n\\n"``.  Empty string when no tables were found.
    page_intent : list[dict]
        Zero or more intent-detection results produced by
        ``PageIntentTagger``.  Each element is an ``IntentResult``
        serialised to a dict via ``dataclasses.asdict``.
    company : str
        BSE ticker / company identifier, e.g. ``"KALYANKJIL"``.
    year : int
        Financial year end, e.g. ``2025`` for FY 2024-25.
    """

    page_number:  int                    = 0
    clean_text:   str                    = ""
    has_table:    bool                   = False
    table_type:   Optional[TableType]    = None
    word_count:   int                    = 0
    is_short:     bool                   = False
    doc_type:     str                    = "ANNUAL_REPORT"
    raw_tables:   str                    = ""
    page_intent:  list                   = field(default_factory=list)
    company:      str                    = ""
    year:         int                    = 0