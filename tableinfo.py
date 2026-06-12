import re

from attr import dataclass
from sqlalchemy import text
@dataclass
class TableExtractor:

    def __init__(self):
        pass

    def is_table_row(self, line):
        return '|' in line.strip()

    def is_separator_row(self, line):
        # e.g. |---|---|  or  --- | ---
        stripped = line.strip()
        return bool(re.fullmatch(r'[\|\s\-:]+', stripped)) and '-' in stripped

    def strip_tables(self, text):
        """
        Splits text into (clean_text, tables_text).
        Detects markdown-style tables: consecutive lines containing '|',
        with a header separator line like |---|---|.
        """
        lines = text.split('\n')
        clean_lines = []
        table_blocks = []
        current_table = []
        in_table = False

    
        i = 0
        while i < len(lines):
            line = lines[i]

            if self.is_table_row(line):
                # lookahead: check if next line is a separator -> confirms table start
                if not in_table:
                    if i + 1 < len(lines) and self.is_separator_row(lines[i + 1]):
                        in_table = True
                        current_table = [line]
                        i += 1
                        continue
                    elif self.is_separator_row(line) and current_table:
                        current_table.append(line)
                        i += 1
                        continue
                    else:
                        clean_lines.append(line)
                else:
                    current_table.append(line)
            else:
                if in_table:
                    # table ended
                    table_blocks.append('\n'.join(current_table))
                    current_table = []
                    in_table = False
                clean_lines.append(line)

            i += 1

        if current_table:
            table_blocks.append('\n'.join(current_table))

        clean_text = '\n'.join(clean_lines)
        tables_text = '\n\n'.join(table_blocks)

        return clean_text, tables_text


    def chunk_tables_by_row(tables_text, context_prefix=""):
        """
        Takes the extracted tables_text (multiple tables separated by blank lines).
        Returns a list of dicts, one per row, with headers prepended.

        Each chunk dict: {
            'table_index': int,
            'row_index': int,
            'text': str,   # embeddable string
            'raw_row': dict # column -> value
        }
        """
        chunks = []
        table_blocks = [t for t in tables_text.split('\n\n') if t.strip()]

        for t_idx, block in enumerate(table_blocks):
            rows = [r for r in block.split('\n') if r.strip()]
            if len(rows) < 2:
                continue

            header_line = rows[0]
            headers = [h.strip() for h in header_line.strip('|').split('|')]

            # rows[1] is the separator (---|---), skip it
            data_rows = rows[2:] if re.fullmatch(r'[\|\s\-:]+', rows[1].strip()) else rows[1:]

            for r_idx, row_line in enumerate(data_rows):
                values = [v.strip() for v in row_line.strip('|').split('|')]
                if len(values) != len(headers):
                    continue  # skip malformed rows

                raw_row = dict(zip(headers, values))

                # Build embeddable text: "Header: Value | Header: Value ..."
                row_text = " | ".join(f"{h}: {v}" for h, v in raw_row.items())
                if context_prefix:
                    row_text = f"{context_prefix} | {row_text}"

                chunks.append({
                    'table_index': t_idx,
                    'row_index': r_idx,
                    'text': row_text,
                    'raw_row': raw_row
                })

        return chunks