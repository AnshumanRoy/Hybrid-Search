"""Generate sample documents (PDF + txt) from data/sample text sources.

Usage:
    python scripts/make_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    text = _read_handbook()
    (SAMPLE_DIR / "acme_handbook.txt").write_text(text, encoding="utf-8")

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            print("PyMuPDF not installed; skipping PDF generation", file=sys.stderr)
            return

    pdf_path = SAMPLE_DIR / "acme_handbook.pdf"
    document = fitz.open()
    chunks = text.split("\n\n")
    per_page = max(3, len(chunks) // 3)
    for i in range(0, len(chunks), per_page):
        page = document.new_page()
        body = chunks[i : i + per_page]
        y = 72
        for block in body:
            page.insert_text((72, y), block, fontsize=11)
            y += 14 * max(1, len(block) // 90 + 2)
            if y > 720:
                break
    document.save(str(pdf_path))
    document.close()
    print(f"wrote {pdf_path}")


def _read_handbook() -> str:
    return (SAMPLE_DIR / "acme_handbook.txt").read_text(encoding="utf-8")


if __name__ == "__main__":
    main()