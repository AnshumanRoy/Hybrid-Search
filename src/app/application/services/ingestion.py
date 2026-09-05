"""Document ingestion orchestration.

Pipeline::

    FileReference ──(bytes)──► Parser ──► ParsedDocument ──► Chunker ──► Chunk[]
                                                                          │
                                                    embedder.embed(texts)  ▼
                                                                       EmbeddedChunk[]
                                                                          │
                              persistence ──(Chunk[])──► repository (SQL) + vector store + bm25

Business rules:
- Content hashing drives incremental ingestion: existing document with the same
  hash ⇒ skip; changed hash ⇒ reprocess (replace); new ⇒ index.
- Processing (parser/chunker/embedder) is separated from persistence: this
  service produces embedded chunks, then hands them to a ``Persister`` so the
  same pipeline is reusable with different persistence backends.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone

from app.common.batching import batched
from app.common.configuration.settings import Settings
from app.common.utils.doc_utils import assert_size_ok
from app.domain.entities import (
    Chunk,
    DocumentRecord,
    EmbeddedChunk,
    FileReference,
    IngestionResult,
    ParsedDocument,
)
from app.domain.exceptions import DocumentParseError, UnsupportedDocumentError
from app.domain.interfaces import Chunker, DocumentParser, Embedder, FileSource, Persister

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        source: FileSource,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        persister: Persister,
        settings: Settings,
    ) -> None:
        self.source = source
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.persister = persister
        self.settings = settings

    def process_dir(self, directory: str | None = None, owner_id: str | None = None) -> list[IngestionResult]:
        """Discover and ingest every file the source reports."""
        results: list[IngestionResult] = []
        for reference in self.source.list_files():
            results.append(self.process_file(reference, owner_id=owner_id))
        return results

    def process_file(
        self,
        reference: FileReference,
        data: bytes | None = None,
        owner_id: str | None = None,
    ) -> IngestionResult:
        started = time.perf_counter()
        try:
            if data is None:
                assert_size_ok(reference, self.settings.max_file_bytes)
                data = self.source.read(reference)
            else:
                reference.size_bytes = len(data)
                assert_size_ok(reference, self.settings.max_file_bytes)
            existing = self._existing_by_hash(data)
            if existing is not None:
                logger.info("document unchanged, skipping; document_id=%s path=%s", existing.document_id, reference.path)
                return IngestionResult(
                    document_id=existing.document_id,
                    filename=reference.name,
                    status="skipped",
                    duration_ms=_ms(started),
                )

            document = self.parser.parse(reference, data)
            chunks = self.chunker.chunk(document)
            if not chunks:
                logger.warning("document yielded no chunks; document_id=%s path=%s", document.document_id, reference.path)
                return IngestionResult(
                    document_id=document.document_id,
                    filename=reference.name,
                    status="failed",
                    error="parsed but produced no chunks",
                    duration_ms=_ms(started),
                )

            embedded = self._embed(chunks, document)
            self.persister.replace_document(document, embedded, owner_id=owner_id)
            logger.info(
                "document indexed; document_id=%s path=%s chunks=%d duration_ms=%.1f",
                document.document_id,
                reference.path,
                len(embedded),
                _ms(started),
            )
            return IngestionResult(
                document_id=document.document_id,
                filename=reference.name,
                status="indexed",
                chunks_embedded=len(embedded),
                duration_ms=_ms(started),
            )
        except (DocumentParseError, UnsupportedDocumentError) as exc:
            logger.warning("document could not be parsed; path=%s error=%s", reference.path, exc)
            return IngestionResult(
                document_id="",
                filename=reference.name,
                status="failed",
                error=str(exc),
                duration_ms=_ms(started),
            )
        except Exception as exc:  # noqa: BLE001 - boundary converts to result
            logger.exception("ingestion failed for path=%s", reference.path)
            return IngestionResult(
                document_id="",
                filename=reference.name,
                status="failed",
                error=str(exc),
                duration_ms=_ms(started),
            )

    def process_bytes(self, filename: str, data: bytes, owner_id: str | None = None) -> IngestionResult:
        """Ingest raw bytes without going through the file source (API path)."""
        reference = FileReference(
            source_id="upload",
            path=filename,
            filename=filename,
            size_bytes=len(data),
            metadata={},
        )
        return self.process_file(reference, data=data, owner_id=owner_id)

    def _embed(self, chunks: list[Chunk], document: ParsedDocument) -> list[EmbeddedChunk]:
        texts = [c.text for c in chunks]
        all_vectors: list[list[float]] = []
        for batch in batched(texts, self.settings.embedding_batch_size, name="ingest_embedding"):
            all_vectors.extend(self.embedder.embed(batch))
        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=all_vectors[i],
                embedding_model=self.embedder.model_name,
                embedding_dimension=self.embedder.dimension,
                embedding_version=self.settings.embedding_version,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _existing_by_hash(self, data: bytes) -> DocumentRecord | None:
        digest = hashlib.sha256(data).hexdigest()
        return self.persister.find_by_hash(digest)

    def ingest_files(self, files: Iterable[FileReference]) -> list[IngestionResult]:
        return [self.process_file(f) for f in files]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)