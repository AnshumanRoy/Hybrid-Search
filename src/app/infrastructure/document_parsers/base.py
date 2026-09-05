"""Base class for document parsers (shared helpers)."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from app.domain.entities import FileReference, ParsedDocument

logger = logging.getLogger(__name__)


class BaseParser:
    _supports: set[str] = set()

    def supported_extensions(self) -> set[str]:
        return set(self._supports)

    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument:
        raise NotImplementedError

    @staticmethod
    def _content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _document_id(reference: FileReference, data: bytes) -> str:
        digest = hashlib.sha256(f"{reference.source_id}::{reference.path}".encode("utf-8")).hexdigest()[:16]
        return f"doc_{digest}"

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)