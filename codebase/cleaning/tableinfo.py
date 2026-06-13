"""
cleaning/tableinfo.py
=====================
Markdown pipe-table detection, extraction, and row-level chunking.

Classes
-------
TableExtractor
    Stateless helper that operates on plain strings.  All methods are
    instance methods so the class can be sub-classed or dependency-injected.

    Public API
    ----------
    strip_tables(text)           → (clean_text: str, tables_text: str)
    chunk_tables_by_row(...)     → list[dict]

Notes
-----
- The original ``chunk_tables_by_row`` was defined without ``self``, making
  it an unbound function masquerading as a method.  It is now a proper
  instance method.
- No external dependencies beyond the standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# TableExtractor
# ---------------------------------------------------------------------------

@dataclass
class TableExtractor:
    """
    Stateless helper for extracting and chunking markdown pipe-tables.

    All public methods are pure functions of their arguments; no internal
    state is mutated between calls, so a single instance may be safely
    reused across pages.
    """

    # ------------------------------------------------------------------
    # Low-level line classifiers
    # ------------------------------------------------------------------

    def is_table_row(self, line: str) -> bool:
        """Return True when *line* contains a pipe character (``|``)."""
        return "|" in line.strip()

    def is_separator_row(self, line: str) -> bool:
        """
        Return True when *line* is a markdown table separator.

        Matches patterns such as ``|---|---|`` or ``--- | ---``.
        """
        stripped = line.strip()
        return bool(re.fullmatch(r"[\|\s\-:]+", stripped)) and "-" in stripped

    # ------------------------------------------------------------------
    # Public: strip_tables
    # ------------------------------------------------------------------

    def strip_tables(self, text: str) -> tuple[str, str]:
        """
        Split *text* into prose and table blocks.

        Scans the text line-by-line.  A table is identified by a data row
        immediately followed by a separator row (``|---|---|``).  Once a
        table starts, all subsequent pipe-rows are consumed until a
        non-pipe line is encountered.

        Parameters
        ----------
        text : str
            Raw markdown text of a single page (after initial cleaning).

        Returns
        -------
        clean_text : str
            The original text with all table blocks removed.
        tables_text : str
            All extracted table blocks joined by ``"\\n\\n"``.
        """
        lines         = text.split("\n")
        clean_lines:  list[str] = []
        table_blocks: list[str] = []
        current_table: list[str] = []
        in_table = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if self.is_table_row(line):
                if not in_table:
                    # Confirm table start: next line must be a separator
                    if i + 1 < len(lines) and self.is_separator_row(lines[i + 1]):
                        in_table = True
                        current_table = [line]
                        i += 1
                        continue
                    elif self.is_separator_row(line) and current_table:
                        # Separator arrived — still part of a building table
                        current_table.append(line)
                        i += 1
                        continue
                    else:
                        # Isolated pipe line — treat as prose
                        clean_lines.append(line)
                else:
                    current_table.append(line)

            else:
                if in_table:
                    # Non-pipe line after table rows → table ended
                    table_blocks.append("\n".join(current_table))
                    current_table = []
                    in_table = False
                clean_lines.append(line)

            i += 1

        # Flush any table still open at EOF
        if current_table:
            table_blocks.append("\n".join(current_table))

        clean_text  = "\n".join(clean_lines)
        tables_text = "\n\n".join(table_blocks)
        return clean_text, tables_text

    # ------------------------------------------------------------------
    # Public: chunk_tables_by_row
    # ------------------------------------------------------------------

    def chunk_tables_by_row(
        self,
        tables_text: str,
        context_prefix: str = "",
    ) -> list[dict]:
        """
        Convert extracted table blocks into one embeddable chunk per row.

        Each data row becomes a self-contained text string of the form::

            "Header1: Value1 | Header2: Value2 | ..."

        An optional *context_prefix* (e.g. the nearest section heading) is
        prepended to every row string to preserve semantic context.

        Parameters
        ----------
        tables_text : str
            One or more pipe-table blocks joined by blank lines, as
            returned by :meth:`strip_tables`.
        context_prefix : str, optional
            Text prepended to every row chunk.  Defaults to ``""``.

        Returns
        -------
        list[dict]
            Each element contains:

            - ``table_index``  (int)  — zero-based index of the source table.
            - ``row_index``    (int)  — zero-based data-row index.
            - ``text``         (str)  — embeddable string for this row.
            - ``raw_row``      (dict) — column-name → cell-value mapping.
        """
        chunks: list[dict] = []
        table_blocks = [t for t in tables_text.split("\n\n") if t.strip()]

        for t_idx, block in enumerate(table_blocks):
            rows = [r for r in block.split("\n") if r.strip()]
            if len(rows) < 2:
                # Need at least a header row + one data row
                continue

            header_line = rows[0]
            headers = [h.strip() for h in header_line.strip("|").split("|")]

            # rows[1] is the separator (---|---); skip it when present
            if re.fullmatch(r"[\|\s\-:]+", rows[1].strip()):
                data_rows = rows[2:]
            else:
                data_rows = rows[1:]

            for r_idx, row_line in enumerate(data_rows):
                values = [v.strip() for v in row_line.strip("|").split("|")]
                if len(values) != len(headers):
                    # Skip malformed rows silently
                    continue

                raw_row = dict(zip(headers, values))

                # "Header: Value | Header: Value ..."
                row_text = " | ".join(f"{h}: {v}" for h, v in raw_row.items())
                if context_prefix:
                    row_text = f"{context_prefix} | {row_text}"

                chunks.append(
                    {
                        "table_index": t_idx,
                        "row_index":   r_idx,
                        "text":        row_text,
                        "raw_row":     raw_row,
                    }
                )

        return chunks