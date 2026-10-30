"""Builds docs/IAA_2026_CST_writeup.pdf (A4, within the 5-page limit) from docs/writeup.md.

The shared doc is the master copy of the write-up: after editing it, export it as Markdown over writeup.md
(keeping the figure and formula blocks) and run

    python3 docs/build_writeup_pdf.py

Markdown -> HTML with python-markdown, HTML -> PDF with headless Chrome; prints the page count.
"""
import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF = HERE / "IAA_2026_CST_writeup.pdf"
CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm;
        @bottom-right { content: counter(page) " / " counter(pages); font: 7.5pt Helvetica, Arial, sans-serif; color: #888; } }
body { font: 10pt/1.36 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #161616; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
.byline { color: #666; margin: 0 0 6pt; }
h2 { font-size: 11.5pt; margin: 10pt 0 3.5pt; padding-bottom: 1.5pt; border-bottom: 0.6pt solid #bbb; break-after: avoid; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0 0 4.5pt; padding-left: 15pt; }
li { margin: 0 0 2pt; }
code { font: 8.4pt Menlo, Consolas, monospace; background: #f2f2f0; padding: 0 1.5pt; border-radius: 2pt; }
table { width: 100%; border-collapse: collapse; margin: 3pt 0 6pt; font-size: 8.7pt; line-height: 1.25; }
th, td { border: 0.5pt solid #c9c9c4; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0efec; font-weight: 600; }
tr { break-inside: avoid; }
figure { margin: 4pt 0 6pt; text-align: center; break-inside: avoid; }
figure img { width: 94%; }
figcaption { font-size: 8.2pt; color: #555; margin-top: 1pt; }
.flow .grid { display: grid; grid-template-columns: 1fr 9pt 1fr 9pt 1fr 9pt 1fr 9pt 1fr; text-align: left;
              font-size: 7.6pt; line-height: 1.25; color: #444; }
.flow .box { border: 0.6pt solid #b9b8b2; border-radius: 3pt; padding: 2.5pt 4pt; }
.flow .box b { display: block; color: #161616; font-size: 7.9pt; margin-bottom: 0.5pt; }
.flow .hl { border: 1pt solid #2f6fd0; background: #e8f0fb; }
.flow .arr { align-self: center; text-align: center; color: #888; }
.flow .down { text-align: center; color: #777; font-size: 7pt; padding: 1.5pt 0; }
.flow figcaption { margin-top: 3pt; }
.formula { margin: 4pt 0 6pt; text-align: center; font-style: italic; }
.frac { display: inline-block; vertical-align: middle; text-align: center; margin-left: 4pt; }
.frac > span { display: block; padding: 0 4pt; }
.num { border-bottom: 0.6pt solid #161616; }
"""


def main():
    body = markdown.markdown((HERE / "writeup.md").read_text(), extensions=["tables"])
    html = HERE / "writeup.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>IAA 2026 write-up, team CST</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    pages = len(PdfReader(PDF).pages)
    print(f"wrote {PDF.name}: {pages} pages")
    return 0 if pages <= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
