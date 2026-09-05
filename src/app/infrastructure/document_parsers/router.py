"""Parser registry: routes an extension to the right ``DocumentParser``."""

from __future__ import annotations

import logging

from app.common.utils.doc_utils import normalize_extension
from app.domain.entities import FileReference, ParsedDocument
from app.domain.exceptions import UnsupportedDocumentError
from app.domain.interfaces import DocumentParser

logger = logging.getLogger(__name__)


class DocumentParserRouter:
    """Implements the ``DocumentParser`` protocol by dispatching on extension."""

    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers
        self._routes: dict[str, DocumentParser] = {}
        for parser in parsers:
            for ext in getattr(parser, "supported_extensions", lambda: set())():
                self._routes[ext] = parser

    def supported_extensions(self) -> set[str]:
        return set(self._routes)

    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument:
        ext = normalize_extension(reference.name)
        parser = self._routes.get(ext)
        if parser is None:
            raise UnsupportedDocumentError(
                f"no parser registered for '.{ext}' (available: {', '.join(sorted(self._routes))})"
            )
        return parser.parse(reference, data)