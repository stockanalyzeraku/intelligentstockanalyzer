# =============================================================================
# CELL 12 — Hierarchical Chunker
# =============================================================================
"""
Create parent, child, table, and atomic-fact chunks from extracted page content.
Chunk IDs follow the pattern: {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import config
from logger import get_logger
from sectiondetector import detect_section, MGMT_SECTIONS
from pdfclassifier import PageContent



@dataclass
class Chunk:
    """
    Represents one unit of text ready for storage in ChromaDB.

    Attributes
    ----------
    id : str
        Unique chunk ID: {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}.
    text : str
        Chunk content.
    collection : str
        Target ChromaDB collection name.
    scrip : str
        Company scrip symbol.
    fiscal_year : str
        Normalised fiscal year (e.g. 'FY25').
    fiscal_year_int : int
        Integer representation (e.g. 2025).
    section : str
        Annual report section name.
    page_number : int
        Source page (1-based).
    chunk_level : str
        'parent', 'child', 'fact', or 'table'.
    content_type : str
        'text', 'table', or 'fact'.
    has_numbers : bool
        True if text contains digit sequences.
    has_percentage : bool
        True if text contains '%'.
    parent_id : str
        ID of the parent chunk ('self' for parent-level chunks).
    source_display : str
        Human-readable source string for citations.
    content_hash : str
        MD5 of normalised text for deduplication.
    """
    id: str
    text: str
    collection: str
    scrip: str
    fiscal_year: str
    fiscal_year_int: int
    section: str
    page_number: int
    chunk_level: str
    content_type: str
    has_numbers: bool
    has_percentage: bool
    parent_id: str
    source_display: str
    content_hash: str

    def __init__(self):
        self._CONFIG = config.Congig.get_instance()

    def to_metadata(self) -> Dict:
        """Return ChromaDB-compatible metadata dict (all values must be str/int/float/bool)."""
        return {
            "scrip": self.scrip,
            "fiscal_year": self.fiscal_year,
            "fiscal_year_int": self.fiscal_year_int,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_level": self.chunk_level,
            "content_type": self.content_type,
            "has_numbers": self.has_numbers,
            "has_percentage": self.has_percentage,
            "parent_id": self.parent_id,
            "source_display": self.source_display,
            "content_hash": self.content_hash,
        }


# ── Fact extraction patterns ────────────────────────────────────────────────
FACT_PATTERNS = [
    re.compile(r"revenue\s+(?:was|is|stood at|of|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:cr|crore|lakh|mn|million|bn|billion)?", re.I),
    re.compile(r"(?:net\s+)?profit\s+(?:was|is|stood at|of|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:cr|crore|lakh|mn|million|bn|billion)?", re.I),
    re.compile(r"ebitda\s+(?:was|is|stood at|of|margin|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?%?", re.I),
    re.compile(r"(?:total\s+)?(?:assets|liabilities|equity)\s+(?:were|was|stood at|of|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:cr|crore|lakh|mn|million|bn|billion)?", re.I),
    re.compile(r"(?:earnings?\s+per\s+share|eps)\s+(?:was|is|of|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?", re.I),
    re.compile(r"dividend\s+(?:of|per\s+share|declared|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?", re.I),
    re.compile(r"(?:market\s+cap(?:itali[sz]ation)?)\s+(?:of|was|is|stood at|:)\s+[₹$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:cr|crore|lakh|mn|million|bn|billion)?", re.I),
    re.compile(r"(?:return\s+on\s+(?:equity|capital|assets)|roe|roce|roa)\s*(?:of|was|is|:)\s+[\d,]+(?:\.\d+)?%?", re.I),
    re.compile(r"(?:debt.to.equity|d/e\s+ratio)\s*(?:of|was|is|stood at|:)\s+[\d,]+(?:\.\d+)?", re.I),
]


def _compute_hash(text: str) -> str:
    """
    Compute MD5 hash of the first 200 normalised characters of text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        8-character hex digest.
    """
    normalised = " ".join(text[:200].lower().split())
    return hashlib.md5(normalised.encode()).hexdigest()[:8]


def _fy_to_int(fy: str) -> int:
    """
    Convert FY string to integer year.

    Parameters
    ----------
    fy : str
        e.g. 'FY25' → 2025, 'FY24' → 2024.

    Returns
    -------
    int
    """
    digits = re.sub(r"[^0-9]", "", fy)
    if len(digits) == 2:
        base = 2000 if int(digits) < 50 else 1900
        return base + int(digits)
    return int(digits)


def _sanitise_id_part(s: str) -> str:
    """Replace non-alphanumeric characters with underscore for use in IDs."""
    return re.sub(r"[^A-Za-z0-9]", "_", s)


def _route_collection(self,chunk_level: str, section: str, content_type: str) -> str:
    """
    Determine the ChromaDB collection for a chunk.

    Parameters
    ----------
    chunk_level : str
        'parent', 'child', 'fact', 'table'.
    section : str
        Section name from detect_section().
    content_type : str
        'text', 'table', 'fact'.

    Returns
    -------
    str
        Collection name constant.
    """
    if chunk_level == "parent":
        return self._CONFIG.COL_PARENT
    if chunk_level == "fact":
        return self._CONFIG.COL_FACTS
    if content_type == "table":
        return self._CONFIG.COL_FACTS
    if section in MGMT_SECTIONS:
        return self._CONFIG.COL_MGMT
    return self._CONFIG.COL_CHILD


def _split_text(text: str, max_tokens: int, overlap: int = 0) -> List[str]:
    """
    Split text into overlapping word-based chunks of approximately max_tokens.

    Parameters
    ----------
    text : str
        Input text.
    max_tokens : int
        Approximate token limit (words used as proxy for tokens).
    overlap : int
        Number of words to overlap between consecutive chunks.

    Returns
    -------
    List[str]
    """
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _extract_facts(self, text: str, scrip: str, fy: str, section: str, page: int, parent_id: str) -> List[Chunk]:
    """
    Extract atomic financial facts from a child chunk using FACT_PATTERNS.

    Parameters
    ----------
    text : str
        Child chunk text.
    scrip : str
        Company scrip.
    fy : str
        Fiscal year string.
    section : str
        Section name.
    page : int
        Page number.
    parent_id : str
        Parent chunk ID.

    Returns
    -------
    List[Chunk]
        Atomic fact chunks destined for COL_FACTS.
    """
    facts: List[Chunk] = []
    fy_int = _fy_to_int(fy)
    for idx, pattern in enumerate(FACT_PATTERNS):
        for match in pattern.finditer(text):
            fact_text = match.group(0).strip()
            if len(fact_text) < 10:
                continue
            fact_id = (
                f"{_sanitise_id_part(scrip)}_{fy}_{_sanitise_id_part(section)}"
                f"_{page}_fact{idx}_{_compute_hash(fact_text)}"
            )
            facts.append(
                Chunk(
                    id=fact_id,
                    text=f"[FACT | {section.upper()} | {fy}]{fact_text}",
                    collection=self._CONFIG.COL_FACTS,
                    scrip=scrip,
                    fiscal_year=fy,
                    fiscal_year_int=fy_int,
                    section=section,
                    page_number=page,
                    chunk_level="fact",
                    content_type="fact",
                    has_numbers=True,
                    has_percentage="%" in fact_text,
                    parent_id=parent_id,
                    source_display=f"{scrip} {fy} p.{page}",
                    content_hash=_compute_hash(fact_text),
                )
            )
    return facts


def create_chunks(
    self,
    pages: List[PageContent],
    tables: List[Dict],
    scrip: str,
    fiscal_year: str,
) -> List[Chunk]:
    """
    Create all chunk types from page content and extracted tables.

    Hierarchy
    ---------
    parent  → 2500-token full section (COL_PARENT)
    child   → 400-token paragraph (COL_CHILD or COL_MGMT by section)
    table   → one table per chunk (COL_FACTS or COL_CHILD)
    fact    → atomic financial fact sentence (COL_FACTS)

    Parameters
    ----------
    pages : List[PageContent]
        Usable page content from classify_and_extract().
    tables : List[Dict]
        Extracted tables from extract_tables().
    scrip : str
        Validated company scrip symbol.
    fiscal_year : str
        Normalised fiscal year (e.g. 'FY25').

    Returns
    -------
    List[Chunk]
    """
    _chunk_logger = get_logger("chunker")
    fy_int = _fy_to_int(fiscal_year)
    all_chunks: List[Chunk] = []

    usable_pages = [p for p in pages if p.is_usable]
    if not usable_pages:
        _chunk_logger.warning("No usable pages found.", scrip=scrip, fy=fiscal_year)
        return []

    # ── Group pages into section buckets ────────────────────────────────────
    # Each bucket: (section_name, list of (page_number, text))
    section_buckets: Dict[str, List[Tuple[int, str]]] = {}
    prev_section = None
    for page in usable_pages:
        section = detect_section(page.text, prev_section)
        prev_section = section
        if section not in section_buckets:
            section_buckets[section] = []
        section_buckets[section].append((page.page_number, page.text))

    # ── Build parent + child chunks per section ──────────────────────────────
    for section, page_tuples in section_buckets.items():
        full_text = "\n\n".join(text for _, text in page_tuples)
        first_page = page_tuples[0][0]

        # Parent chunk
        parent_id = (
            f"{_sanitise_id_part(scrip)}_{fiscal_year}"
            f"_{_sanitise_id_part(section)}_{first_page}_parent"
        )
        all_chunks.append(
            Chunk(
                id=parent_id,
                text=full_text[: self._CONFIG.PARENT_CHUNK_TOKENS * 5],  # ~5 chars/token
                collection=self._CONFIG.COL_PARENT,
                scrip=scrip,
                fiscal_year=fiscal_year,
                fiscal_year_int=fy_int,
                section=section,
                page_number=first_page,
                chunk_level="parent",
                content_type="text",
                has_numbers=bool(re.search(r"\d", full_text)),
                has_percentage="%" in full_text,
                parent_id=parent_id,
                source_display=f"{scrip} {fiscal_year} {section} p.{first_page}",
                content_hash=_compute_hash(full_text),
            )
        )

        # Child chunks
        child_splits = _split_text(full_text, self._CONFIG.CHILD_CHUNK_TOKENS, self._CONFIG.CHUNK_OVERLAP_TOKENS)
        for child_idx, child_text in enumerate(child_splits):
            page_num = first_page  # approximate; pages blend within a section
            child_id = (
                f"{_sanitise_id_part(scrip)}_{fiscal_year}"
                f"_{_sanitise_id_part(section)}_{page_num}_{child_idx}"
            )
            col = _route_collection("child", section, "text")
            child_chunk = Chunk(
                id=child_id,
                text=child_text,
                collection=col,
                scrip=scrip,
                fiscal_year=fiscal_year,
                fiscal_year_int=fy_int,
                section=section,
                page_number=page_num,
                chunk_level="child",
                content_type="text",
                has_numbers=bool(re.search(r"\d", child_text)),
                has_percentage="%" in child_text,
                parent_id=parent_id,
                source_display=f"{scrip} {fiscal_year} {section} p.{page_num}",
                content_hash=_compute_hash(child_text),
            )
            all_chunks.append(child_chunk)
            # Extract atomic facts from child chunks
            all_chunks.extend(
                _extract_facts(child_text, scrip, fiscal_year, section, page_num, parent_id)
            )

    # ── Table chunks ──────────────────────────────────────────────────────────
    table_page_set = {p.page_number for p in usable_pages}
    for tbl_idx, tbl in enumerate(tables):
        tbl_page = tbl.get("page_number", 0)
        if tbl_page not in table_page_set:
            continue  # skip tables from scanned pages
        tbl_text = tbl.get("text", "")
        if not tbl_text.strip():
            continue
        # Detect section from table's page text
        matching_pages = [p for p in usable_pages if p.page_number == tbl_page]
        tbl_section = detect_section(
            matching_pages[0].text if matching_pages else "", prev_section
        )
        tbl_id = (
            f"{_sanitise_id_part(scrip)}_{fiscal_year}"
            f"_{_sanitise_id_part(tbl_section)}_{tbl_page}_tbl{tbl_idx}"
        )
        tbl_parent_id = (
            f"{_sanitise_id_part(scrip)}_{fiscal_year}"
            f"_{_sanitise_id_part(tbl_section)}_{tbl_page}_parent"
        )
        tbl_col = self._CONFIG.COL_FACTS if tbl.get("is_financial") else self._CONFIG.COL_CHILD
        all_chunks.append(
            Chunk(
                id=tbl_id,
                text=tbl_text,
                collection=tbl_col,
                scrip=scrip,
                fiscal_year=fiscal_year,
                fiscal_year_int=fy_int,
                section=tbl_section,
                page_number=tbl_page,
                chunk_level="child",
                content_type="table",
                has_numbers=bool(re.search(r"\d", tbl_text)),
                has_percentage="%" in tbl_text,
                parent_id=tbl_parent_id,
                source_display=f"{scrip} {fiscal_year} {tbl_section} p.{tbl_page} [table]",
                content_hash=_compute_hash(tbl_text),
            )
        )

    _chunk_logger.info(
        "Chunking complete.",
        scrip=scrip,
        fy=fiscal_year,
        total_chunks=len(all_chunks),
    )
    return all_chunks

# ----------------------------------------------------------------------------
# Cell 12: Hierarchical Chunker
# Purpose: Convert pages + tables into parent/child/fact/table Chunk objects.
# Key Classes: Chunk (dataclass)
# Key Functions:
#   create_chunks(pages, tables, scrip, fiscal_year) → List[Chunk]
#   _extract_facts(text, scrip, fy, section, page, parent_id) → List[Chunk]
#   _route_collection(chunk_level, section, content_type) → str
#   _split_text(text, max_tokens, overlap) → List[str]
#   _compute_hash(text) → str
# Key Constants/Config: FACT_PATTERNS (9 patterns), CONFIG.PARENT_CHUNK_TOKENS,
#   CONFIG.CHILD_CHUNK_TOKENS, CONFIG.CHUNK_OVERLAP_TOKENS
# Imports exported: Chunk, create_chunks, FACT_PATTERNS
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 9 (PageContent),
#   Cell 10 (detect_section, MGMT_SECTIONS)
# Critical notes: Chunk.id format is {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}.
#   parent_id on child and fact chunks links back to parent for context expansion.
#   Tables from scanned pages are skipped.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
