"""Convert CODE_REVIEW_REPORT.txt to a paginated PDF (fpdf2). ASCII input only."""
import textwrap
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "CODE_REVIEW_REPORT.txt"
DST = ROOT / "CODE_REVIEW_REPORT.pdf"

WIDTH = 108  # chars per line at 8.5pt Courier


class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Courier", "I", 7)
        self.cell(0, 6, "Jarvis code review - page %s" % self.page_no(), align="C")


pdf = PDF(format="Letter")
pdf.set_auto_page_break(auto=True, margin=14)
pdf.add_page()
for raw in SRC.read_text(encoding="utf-8").splitlines():
    line = raw.rstrip().encode("latin-1", "replace").decode("latin-1")
    is_head = line.startswith(("== ", "SECTION ", "FINDING "))
    pdf.set_font("Courier", "B" if is_head else "", 8.5)
    if not line:
        pdf.ln(3)
        continue
    for piece in textwrap.wrap(line, WIDTH) or [""]:
        pdf.multi_cell(0, 4.4, piece, new_x="LMARGIN", new_y="NEXT")

pdf.output(str(DST))
print("wrote", DST, DST.stat().st_size, "bytes,", pdf.page_no(), "pages")
