"""Plain-text/markdown parser (name → single "page")."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from app.domain.entities import DocumentPage, FileReference, ParsedDocument
from app.domain.exceptions import DocumentParseError
from app.infrastructure.document_parsers.base import BaseParser

logger = logging.getLogger(__name__)

_MAX_TEXT_BYTES = 5 * 1024 * 1024


class TextParser(BaseParser):
    _supports = {"txt", "md", "markdown"}

    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument:
        if len(data) > _MAX_TEXT_BYTES:
            raise DocumentParseError(f"text file exceeds {_MAX_TEXT_BYTES} bytes limit")
        if b"\x00" in data:
            raise DocumentParseError(f"'{reference.name}' appears to be binary; not valid text")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise DocumentParseError(f"'{reference.name}' is not valid UTF-8 text") from None

        normalized = "\n".join(line.rstrip() for line in text.split("\n")).strip()
        if not normalized:
            raise DocumentParseError(f"'{reference.name}' contains no text")

        return ParsedDocument(
            document_id=self._document_id(reference, data),
            filename=reference.name,
            source_uri=reference.uri,
            source_id=reference.source_id,
            content_hash=self._content_hash(data),
            page_count=1,
            pages=[DocumentPage(page_number=1, text=normalized, metadata={"is_empty": False})],
            metadata={"page_count": 1},
            created_at=self._utcnow(),
        )