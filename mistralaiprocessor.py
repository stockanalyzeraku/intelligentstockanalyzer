import os
import re
from pathlib import Path
from mistralai.client import Mistral
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import json
from dataclasses import asdict, dataclass
from typing import List

from logger import get_logger
from inputvalidator import InputValidator
from config import CONFIG

import fitz
@dataclass
class PageContent:
    """
    Content extracted from one page of PDF
    """
    page_num: int
    text: str

class MistralAIProcessor:
    """
    End-to-end processor that:
      1. Uploads a PDF to Mistral and runs OCR.
      2. Saves the raw markdown output as a backup.
      3. Converts the markdown into a styled ReportLab PDF.
    """

    def __init__(self, api_key: str, pdf_path: str, output_file: str):
        """
        Parameters
        ----------
        api_key     : Mistral API key.
        pdf_path    : Absolute path to the source PDF.
        output_file : Absolute path for the generated output PDF.
        """
        
        self.log = get_logger("MistralAI")
        self.log.info("Initialising MistralAIProcessor")

        # Validate inputs before storing them
        self._api_key = api_key or CONFIG.MISTRAL_API_KEY
        validate = InputValidator()
        self._pdf_path    = validate.validate_pdf_path(pdf_path)
        self._output_file = output_file
        self._pages: list[PageContent] = []

        self._client        : Mistral | None = None
        self._markdown_text : str            = ""
        self._story         : list           = []

        self.log.info(
            f"Config — source: '{self._pdf_path}' | output: '{self._output_file}'"
        )


    def run(self) -> str:           #Function to run Process
        """
        Full pipeline: OCR  →  markdown backup  →  styled PDF.

        Returns
        -------
        str : Path to the generated output PDF.
        """
        self.log.info("Pipeline started")
        self._init_client()
        self._upload_pdf_and_process_ocr()
    #    self._save_markdown_backup()
    #    self._build_pdf()
        self.log.info(f"Pipeline complete — output: '{self._output_file}'")
        return self._output_file

    def save_pages_to_json(self, pages: list[PageContent], output_path: str) -> None:       #Save pages to JSON
        """
        Saves a list of PageContent objects to a JSON file.
        """

        data = [asdict(pc) for pc in pages]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def _init_client(self) -> None:         #Initalize Client
        """Creates and stores the authenticated Mistral client."""
    
        self.log.info("Creating Mistral client")
        self._client = Mistral(api_key=self._api_key) 
        self.log.info("Mistral client ready")

    def _get_signed_url(self, file_id: str) -> str:
        """
        Retrieves a temporary signed URL for an uploaded file.

        Parameters
        ----------
        file_id : The Mistral file ID returned by :meth:`_upload_pdf`.

        Returns
        -------
        str : Signed download URL.
        """
        self.log.info(f"Fetching signed URL for file_id: {file_id}")
        signed = self._client.files.get_signed_url(file_id=file_id)
        self.log.info("Signed URL obtained")
        return signed.url
    
    def _upload_pdf_and_process_ocr(self) -> str:
        """
        Uploads the source PDF to Mistral, obtain signed URL and get page by page output from OCR.

        Returns
        -------
        str : The uploaded file's ID.
        """
        self.log.info(f"Uploading '{Path(self._pdf_path).name}' to Mistral")
        doc = fitz.open(self._pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix       = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            uploaded = self._client.files.upload(
                file={"file_name": os.path.basename(self._pdf_path), "content": img_bytes},
                purpose="ocr")
            self.log.info(f"Upload complete — file_id: {uploaded.id}")
            sign_url = self._get_signed_url(uploaded.id)
            self.log.info(f"Processing file — file_id: {uploaded.id} page_number: {page_num}")
            self._process_ocr(sign_url,page_num)    
        self.log.info(f"Processed file — file_id: {uploaded.id}")
       
        self.log.info(f"JSON Dumping file — file_id: {uploaded.id}")
        self.save_pages_to_json(self._pages, os.path.join(CONFIG.UPLOADS_PATH,"KALYANKJIL","ANNUAL_2025","Kalyan.json"))

        return uploaded.id
    
    def _process_ocr(self, document_url: str, page_num: int) -> None:
        """
        Calls the Mistral OCR endpoint and stores combined markdown.

        Parameters
        ----------
        document_url : Signed URL of the uploaded PDF.
        """
        self.log.info(f"Running OCR with model '{CONFIG.MISTRAL_MODEL_OCR}'")
        ocr_response = self._client.ocr.process(
            model=CONFIG.MISTRAL_MODEL_OCR,
            document={"type": "document_url", "document_url": document_url},
            include_image_base64=False,
        )
        self._pages.append(
            PageContent(
                page_num=page_num + 1,
                text=ocr_response.pages[0].markdown
        ))

        self._markdown_text = "\n\n---\n\n".join(
            page.markdown for page in ocr_response.pages
        )
        self.log.info(f"OCR complete — pages processed: {len(ocr_response.pages)}")

    # ------------------------------------------------------------------ #
    #  Step 2 – Markdown backup                                            #
    # ------------------------------------------------------------------ #

    def _save_markdown_backup(self) -> None:
        """Writes raw OCR markdown to a .txt file beside the output PDF."""
        md_path = self._output_file.replace(".pdf", ".txt")
        self.log.info(f"Saving markdown backup to '{md_path}'")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(self._markdown_text)
        self.log.info("Markdown backup saved")

    # ------------------------------------------------------------------ #
    #  Step 3 – PDF generation                                             #
    # ------------------------------------------------------------------ #

    def _build_pdf(self) -> None:
        """Parses stored markdown, builds a ReportLab story, writes PDF."""
        self.log.info("Building PDF from markdown")
        self._story = self._parse_markdown_to_story(self._markdown_text)
        self.log.info(f"Story built — {len(self._story)} elements")
        self._write_pdf()

    def _write_pdf(self) -> None:
        """Constructs the ReportLab document and saves the PDF to disk."""
        self.log.info(f"Writing PDF to '{self._output_file}'")
        doc = SimpleDocTemplate(
            self._output_file,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        doc.build(self._story)
        self.log.info("PDF written successfully")

    # ------------------------------------------------------------------ #
    #  Markdown → ReportLab story                                          #
    # ------------------------------------------------------------------ #

    def _parse_markdown_to_story(self, md: str) -> list:
        """
        Converts a markdown string into a list of ReportLab flowables.

        Parameters
        ----------
        md : Raw markdown text (may contain headings, tables, bullets, rules).

        Returns
        -------
        list : ReportLab flowable objects ready for ``doc.build()``.
        """
        styles = self._build_styles()
        story  = []
        lines  = md.split("\n")
        i      = 0

        while i < len(lines):
            line = lines[i]

            # ── Headings ──────────────────────────────────────────────
            if line.startswith("### "):
                story.append(Paragraph(self._escape(line[4:].strip()), styles["H3"]))

            elif line.startswith("## "):
                story.append(Paragraph(self._escape(line[3:].strip()), styles["H2"]))

            elif line.startswith("# "):
                story.append(Paragraph(self._escape(line[2:].strip()), styles["H1"]))

            # ── Tables ────────────────────────────────────────────────
            elif line.startswith("|"):
                table_flowable, i = self._parse_table(lines, i, styles)
                if table_flowable:
                    story.append(table_flowable)
                    story.append(Spacer(1, 4))
                continue  # i already advanced inside _parse_table

            # ── Bullet points ─────────────────────────────────────────
            elif re.match(r"^[-*] ", line):
                story.append(
                    Paragraph(
                        "• " + self._escape(line[2:].strip()),
                        styles["BulletItem"],
                    )
                )

            # ── Horizontal rules → page break ─────────────────────────
            elif line.strip() in ("---", "***", "___"):
                story.append(PageBreak())

            # ── Blank lines ───────────────────────────────────────────
            elif line.strip() == "":
                story.append(Spacer(1, 4))

            # ── Regular body text ─────────────────────────────────────
            else:
                story.append(Paragraph(self._escape(line.strip()), styles["Body"]))

            i += 1

        return story

    def _parse_table(self, lines: list, start: int, styles: dict) -> tuple:
        """
        Reads consecutive pipe-delimited lines and returns a Table flowable.

        Parameters
        ----------
        lines : All lines of the markdown document.
        start : Index of the first ``|`` line.
        styles: Paragraph styles (unused here but kept for consistency).

        Returns
        -------
        tuple[Table | None, int] : The Table flowable (or None) and the
                                   updated line index.
        """
        i           = start
        table_lines = []

        while i < len(lines) and lines[i].startswith("|"):
            row_text = lines[i]
            # Skip markdown separator rows like |---|---|
            if not re.match(r"^\|[-| :]+\|$", row_text):
                cells = [
                    self._escape(c.strip())
                    for c in row_text.strip("|").split("|")
                ]
                table_lines.append(cells)
            i += 1

        if not table_lines:
            return None, i

        ncols     = max(len(r) for r in table_lines)
        data      = [r + [""] * (ncols - len(r)) for r in table_lines]
        col_width = (A4[0] - 40 * mm) / ncols

        tbl = Table(data, colWidths=[col_width] * ncols, repeatRows=1)
        tbl.setStyle(
            TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#003366")),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        return tbl, i

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _escape(text: str) -> str:
        """Escapes HTML special characters for safe use in ReportLab paragraphs."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _build_styles() -> dict:
        """
        Creates and returns a dictionary of named ReportLab ParagraphStyles.

        Returns
        -------
        dict : Keys are style names; values are ParagraphStyle objects.
        """
        base   = getSampleStyleSheet()
        custom = {
            "H1": ParagraphStyle(
                name="H1", parent=base["Heading1"],
                fontSize=16, spaceAfter=8,
            ),
            "H2": ParagraphStyle(
                name="H2", parent=base["Heading2"],
                fontSize=13, spaceAfter=6,
            ),
            "H3": ParagraphStyle(
                name="H3", parent=base["Heading3"],
                fontSize=11, spaceAfter=4,
            ),
            "Body": ParagraphStyle(
                name="Body", parent=base["Normal"],
                fontSize=9, spaceAfter=4, leading=13,
            ),
            "BulletItem": ParagraphStyle(
                name="BulletItem", parent=base["Normal"],
                fontSize=9, leftIndent=12, bulletIndent=4, spaceAfter=2,
            ),
        }
        return custom


# ────────────────────────────────────────────────────────────────────────────
# Entry point (quick smoke-test / CLI usage)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processor = MistralAIProcessor(
        api_key="XLmWO4UBEzyVEbYxCMsVKKCEf7jw55mD",
        pdf_path=os.path.join(CONFIG.UPLOADS_PATH,"KALYANKJIL","ANNUAL_2025","KALYANKJIL_ANNUAL_2025.pdf"),
        output_file=os.path.join(CONFIG.UPLOADS_PATH,"KALYANKJIL","ANNUAL_2025","KALYAN_ANNUAL_MI_2025.pdf"),
    )
    output_path = processor.run()
    print(f"Done → {output_path}")