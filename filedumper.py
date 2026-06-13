import os
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

if __name__ == "__main__":
    files = [
        "config.py",
        "tableinfo.py",
        "textcleaner.py",
        "pageintent.py",
        "cleanresult.py",
        "mistralaiprocessor.py",
        "embeddingprepared.py"       
]

    # with open("all_code.md", "w", encoding="utf-8") as out:
    #     for file in files:
    #         out.write(f"{'='*60}\n")
    #         out.write(f"FILE: {file}\n")
    #         out.write(f"{'='*60}\n\n")
    #         with open(file, "r", encoding="utf-8") as fh:
    #             out.write(fh.read())
    #         out.write("\n\n")

    # print("Done! Saved to all_code.txt")
    doc    = SimpleDocTemplate("all_code.pdf", pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle("Code", fontName="Courier", fontSize=7, leading=10)
    head_style = ParagraphStyle("Head", fontName="Helvetica-Bold", fontSize=12, spaceAfter=6)

    story = []

    for file in files:
    # File heading
        story.append(Paragraph(f"FILE: {file}", head_style))
        story.append(Spacer(1, 4))

    # File contents
        with open(file, "r", encoding="utf-8") as fh:
            for line in fh.readlines():
                line = (line.rstrip()
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
                story.append(Paragraph(line or " ", code_style))

        story.append(Spacer(1, 20))

    doc.build(story)
    print("Done! Saved to all_code.pdf")