"""PyMuPDF-based PDF parser.

- Extracts one normalized text block per page (``\\n``-joined lines, trimmed).
- Preserves page numbers and useful per-page metadata.
- Gracefully handles empty pages (keeps an empty text slot with ``is_empty``).
- Raises ``DocumentParseError`` for encrypted, non-PDF, or exhausted documents
  instead of letting raw exceptions bubble up.

The heavy ``fitz`` import is delayed until the first parse call so importing
the rest of the pipeline never pulls PyMuPDF into memory.
"""

from __future__ import annotations

import logging

from app.common.configuration.settings import Settings
from app.domain.entities import DocumentPage, FileReference, ParsedDocument
from app.domain.exceptions import DocumentParseError
from app.infrastructure.document_parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    def __init__(self, settings: Settings) -> None:
        self.max_pages = settings.max_pdf_pages
        self.max_file_bytes = settings.max_file_bytes

    _supports = {"pdf"}

    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument:
        try:
            import pymupdf as fitz  # PyMuPDF (lazy; preferred import name)
        except ImportError:
            try:
                import fitz  # legacy top-level module
            except ImportError as exc:  # pragma: no cover - env with missing optional dep
                raise DocumentParseError("PyMuPDF is not installed; cannot parse PDFs") from exc

        if data[:5] != b"%PDF-":
            raise DocumentParseError(f"'{reference.name}' is not a PDF document (bad magic header)")

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError(f"failed to open PDF '{reference.name}': {exc}") from exc

        if document.needs_pass:
            document.close()
            raise DocumentParseError(f"PDF '{reference.name}' is password-protected")

        total_pages = document.page_count
        page_count = min(total_pages, self.max_pages)
        parsed_pages: list[DocumentPage] = []

        try:
            for page_index in range(page_count):
                page = document.load_page(page_index)
                raw_text = page.get_text("text", flags=2) or ""
                text = self._normalize(raw_text)
                size = page.rect
                parsed_pages.append(
                    DocumentPage(
                        page_number=page_index + 1,
                        text=text,
                        metadata={
                            "is_empty": not text.strip(),
                            "width": round(size.width, 2),
                            "height": round(size.height, 2),
                            "rotation": page.rotation,
                        },
                    )
                )
        except Exception as exc:
            raise DocumentParseError(f"failed to extract text from '{reference.name}': {exc}") from exc
        finally:
            document.close()

        if not parsed_pages:
            raise DocumentParseError(f"PDF '{reference.name}' contains no readable pages")

        return ParsedDocument(
            document_id=self._document_id(reference, data),
            filename=reference.name,
            source_uri=reference.uri,
            source_id=reference.source_id,
            content_hash=self._content_hash(data),
            page_count=len(parsed_pages),
            pages=parsed_pages,
            metadata={
                "page_count": len(parsed_pages),
                "max_pages_exceeded": total_pages > self.max_pages,
            },
            created_at=self._utcnow(),
        )

    @staticmethod
    def _normalize(raw: str) -> str:
        lines = [line.strip() for line in raw.split("\n")]
        return "\n".join(lines).strip()