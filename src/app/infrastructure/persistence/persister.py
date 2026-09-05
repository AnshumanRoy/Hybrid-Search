"""Persistence coordinator implementing the ingestion ``Persister`` protocol.

Coordinates the SQL repository, the vector store and the lexical index so the
ingestion service never touches infrastructure directly.  Replacing a document
removes its old chunks from *all* backends before inserting new ones.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.domain.entities import DocumentRecord, EmbeddedChunk, ParsedDocument
from app.domain.exceptions import VectorStoreError
from app.domain.interfaces import DocumentRepository, LexicalRetriever, VectorStore

logger = logging.getLogger(__name__)


class IndexPersister:
    def __init__(
        self,
        repository: DocumentRepository,
        vector_store: VectorStore,
        lexical: LexicalRetriever,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.lexical = lexical

    def find_by_hash(self, content_hash: str) -> DocumentRecord | None:
        return self.repository.get_document_by_hash(content_hash)

    def persist_document(
        self,
        document: ParsedDocument,
        embedded_chunks: list[EmbeddedChunk],
        status: str = "indexed",
    ) -> None:
        now = datetime.now(timezone.utc)
        self.repository.upsert_document(
            DocumentRecord(
                document_id=document.document_id,
                filename=document.filename,
                source_uri=document.source_uri,
                source_id=document.source_id,
                content_hash=document.content_hash,
                metadata=document.metadata,
                page_count=document.page_count,
                status=status,
                created_at=document.created_at,
                updated_at=now,
            )
        )
        self.repository.add_chunks([ec.chunk for ec in embedded_chunks])
        self.vector_store.upsert(embedded_chunks)
        self.lexical.add_documents([ec.chunk for ec in embedded_chunks])

    def replace_document(
        self,
        document: ParsedDocument,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        logger.debug("replacing document document_id=%s", document.document_id)
        self._clear_document(document.document_id)
        self.persist_document(document, embedded_chunks, status="indexed")

    def delete_document(self, document_id: str) -> None:
        self._clear_document(document_id)
        self.repository.delete_document(document_id)

    def _clear_document(self, document_id: str) -> None:
        chunks = self.repository.get_chunks_by_document(document_id)
        chunk_ids = [c.chunk_id for c in chunks]
        try:
            self.vector_store.delete(chunk_ids)
            self.lexical.remove_documents(chunk_ids)
        except VectorStoreError:
            logger.warning("vector-store cleanup failed for document_id=%s", document_id, exc_info=True)
        self.repository.delete_chunks_by_document(document_id)