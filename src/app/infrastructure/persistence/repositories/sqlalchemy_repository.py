"""SQLAlchemy-backed document/chunk repository (implements the domain Protocol)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy.orm import Session, sessionmaker

from app.domain.entities import Chunk, ChunkRecord, DocumentRecord
from app.domain.interfaces import DocumentRepository
from app.infrastructure.persistence.models import ChunkRow, DocumentRow

logger = logging.getLogger(__name__)


class SQLAlchemyDocumentRepository(DocumentRepository):
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._session_factory() as session:
            row = session.get(DocumentRow, document_id)
            return self._to_document(row) if row else None

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._session_factory() as session:
            row = (
                session.query(DocumentRow)
                .filter(DocumentRow.content_hash == content_hash)
                .order_by(DocumentRow.updated_at.desc())
                .first()
            )
            return self._to_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._session_factory() as session:
            rows = session.query(DocumentRow).order_by(DocumentRow.created_at.desc()).all()
            return [self._to_document(r) for r in rows]

    def upsert_document(self, record: DocumentRecord) -> DocumentRecord:
        with self._session_factory() as session:
            row = session.get(DocumentRow, record.document_id)
            if row is None:
                row = DocumentRow(document_id=record.document_id)
                session.add(row)
            row.filename = record.filename
            row.source_uri = record.source_uri
            row.source_id = record.source_id
            row.content_hash = record.content_hash
            row.metadata_json = record.metadata
            row.page_count = record.page_count
            row.status = record.status
            row.created_at = record.created_at
            row.updated_at = record.updated_at
            session.commit()
            return record

    def delete_document(self, document_id: str) -> None:
        with self._session_factory() as session:
            row = session.get(DocumentRow, document_id)
            if row:
                session.delete(row)
                session.commit()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self._session_factory() as session:
            for chunk in chunks:
                session.merge(
                    ChunkRow(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        source_uri=chunk.source_uri,
                        page_number=chunk.page_number,
                        text=chunk.text,
                        chunk_index=chunk.chunk_index,
                        metadata_json=chunk.metadata,
                    )
                )
            session.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._session_factory() as session:
            row = session.get(ChunkRow, chunk_id)
            return self._to_chunk(row) if row else None

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        with self._session_factory() as session:
            rows = session.query(ChunkRow).filter(ChunkRow.document_id == document_id).all()
            return [self._to_chunk(r) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        with self._session_factory() as session:
            if not chunk_ids:
                return {}
            rows = session.query(ChunkRow).filter(ChunkRow.chunk_id.in_(list(chunk_ids))).all()
            return {r.chunk_id: self._to_chunk(r) for r in rows}

    def delete_chunks_by_document(self, document_id: str) -> None:
        with self._session_factory() as session:
            session.query(ChunkRow).filter(ChunkRow.document_id == document_id).delete()
            session.commit()

    @staticmethod
    def _to_document(row: DocumentRow) -> DocumentRecord:
        return DocumentRecord(
            document_id=row.document_id,
            filename=row.filename,
            source_uri=row.source_uri,
            source_id=row.source_id,
            content_hash=row.content_hash,
            metadata=row.metadata_json or {},
            page_count=row.page_count,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_chunk(row: ChunkRow) -> Chunk:
        return Chunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            filename=row.filename,
            source_uri=row.source_uri,
            page_number=row.page_number,
            text=row.text,
            chunk_index=row.chunk_index,
            metadata=row.metadata_json or {},
        )