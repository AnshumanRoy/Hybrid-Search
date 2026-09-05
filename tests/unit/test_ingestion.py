"""Hash-based incremental ingestion tests (skip unchanged / reprocess changed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp

from app.application.services.ingestion import IngestionService
from app.common.configuration.settings import Settings
from app.domain.entities import DocumentPage, FileReference, ParsedDocument
from app.domain.interfaces import Chunker, DocumentParser
from tests.fakes import FakeEmbedder, make_chunk


@dataclass
class RecordingPersister:
    """Records every call the persister receives."""

    calls: list[str] = field(default_factory=list)
    stored_hash: str | None = None
    stored_doc_id: str | None = None

    def find_by_hash(self, content_hash: str) -> object:
        if content_hash == self.stored_hash:
            from datetime import datetime, timezone

            from app.domain.entities import DocumentRecord

            return DocumentRecord(
                document_id=self.stored_doc_id or "doc_x",
                filename="file.txt",
                content_hash=content_hash,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        return None

    def persist_document(self, document, embedded_chunks, status="indexed", owner_id=None) -> None:
        self.calls.append(f"persist:{document.document_id}")
        self.stored_hash = document.content_hash
        self.stored_doc_id = document.document_id

    def replace_document(self, document, embedded_chunks, owner_id=None) -> None:
        self.calls.append(f"replace:{document.document_id}")
        self.stored_hash = document.content_hash
        self.stored_doc_id = document.document_id

    def delete_document(self, document_id) -> None:
        self.calls.append(f"delete:{document_id}")


class FakeSource:
    def __init__(self, files: list[tuple[str, bytes]]) -> None:
        self.files = files

    def list_files(self):
        for name, data in self.files:
            yield FileReference(path=name, filename=name, size_bytes=len(data))

    def read(self, file: FileReference) -> bytes:
        for name, data in self.files:
            if name == file.path:
                return data
        raise FileNotFoundError(file.path)


class ByteParser(DocumentParser):
    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument:
        import hashlib

        return ParsedDocument(
            document_id=f"doc_{reference.path}",
            filename=reference.name,
            source_uri=reference.uri,
            content_hash=hashlib.sha256(data).hexdigest(),
            page_count=1,
            pages=[DocumentPage(page_number=1, text=data.decode("utf-8"))],
            metadata={"page_count": 1},
        )


class FlatChunker(Chunker):
    def chunk(self, document: ParsedDocument):
        return [
            make_chunk(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}::c0",
                text=document.pages[0].text,
                page=1,
                index=0,
            )
        ]


def _service(data_bytes: bytes, persister: RecordingPersister) -> IngestionService:
    settings = Settings(
        embedding_model="fake-model",
        vector_dimension=8,
        embedding_batch_size=4,
        database_url="sqlite://",
        data_dir=Path(mkdtemp()),
        cache_enabled=False,
    )
    source = FakeSource([("doc.txt", data_bytes)])
    return IngestionService(source, ByteParser(), FlatChunker(), FakeEmbedder(), persister, settings)


def _tmp() -> Path:
    return Path(mkdtemp())


def test_new_document_is_indexed() -> None:
    persister = RecordingPersister()
    service = _service(b"fresh content v1", persister)
    result = service.process_file(next(iter(service.source.list_files())))
    assert result.status == "indexed"
    assert persister.calls == ["replace:doc_doc.txt"], persister.calls


def test_unchanged_document_skipped() -> None:
    persister = RecordingPersister()
    service = _service(b"stable content", persister)
    ref = next(iter(service.source.list_files()))
    first = service.process_file(ref)
    assert first.status == "indexed"
    persister.calls.clear()
    second = service.process_file(ref)
    assert second.status == "skipped"
    assert persister.calls == []


def test_changed_document_reprocessed() -> None:
    persister = RecordingPersister()
    service = _service(b"version one", persister)
    ref = next(iter(service.source.list_files()))
    assert service.process_file(ref).status == "indexed"

    service2 = _service(b"version two changed", persister)
    result = service2.process_file(next(iter(service2.source.list_files())))
    assert result.status == "indexed"
    assert persister.calls == ["replace:doc_doc.txt", "replace:doc_doc.txt"]


def test_process_file_returns_failed_on_error() -> None:
    persister = RecordingPersister()
    service = _service(b"x", persister)

    class BoomParser(DocumentParser):
        def parse(self, reference, data):
            raise ValueError("boom")

    service.parser = BoomParser()
    ref = next(iter(service.source.list_files()))
    result = service.process_file(ref)
    assert result.status == "failed"
    assert "boom" in result.error