"""DocumentService access-rule tests (service layer, in-memory SQL, no network)."""

from __future__ import annotations

import pytest

from app.application.services.document_service import DocumentService
from app.application.services.ingestion import IngestionService
from app.common.configuration.settings import Settings
from app.domain.entities import DocumentRecord, SharingPermission, User
from app.domain.exceptions import DocumentNotFoundError, PermissionDeniedError
from app.domain.interfaces import FileSource
from app.infrastructure.chunkers.sentence import SentenceAwareChunker
from app.infrastructure.document_parsers.text import TextParser
from app.infrastructure.persistence.models import init_db, make_engine, session_factory_for
from app.infrastructure.persistence.persister import IndexPersister
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository
from app.infrastructure.persistence.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository
from app.infrastructure.retrievers.bm25 import BM25Retriever
from tests.fakes import FakeEmbedder, FakeVectorStore


class _NoSource(FileSource):
    def list_files(self):
        return []

    def read(self, file):  # pragma: no cover - never called via process_bytes
        raise AssertionError("process_bytes must not read from the source")


@pytest.fixture
def service():
    engine = make_engine("sqlite://", in_memory=True)
    init_db(engine)
    factory = session_factory_for(engine)
    users = SQLAlchemyUserRepository(factory)
    repo = SQLAlchemyDocumentRepository(factory)
    settings = Settings(
        embedding_model="fake-service-model",
        vector_dimension=8,
        embedding_batch_size=4,
        chunk_size=120,
        chunk_overlap=20,
    )
    embedder = FakeEmbedder(model_name="fake-service-model", dimension=8)
    persister = IndexPersister(repo, FakeVectorStore(8), BM25Retriever())
    ingestion = IngestionService(
        source=_NoSource(),
        parser=TextParser(),
        chunker=SentenceAwareChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        persister=persister,
        settings=settings,
    )
    return DocumentService(ingestion, repo, users), repo, users


def _make_user(users, email: str) -> User:
    return users.create_user(email, "x" * 16, display_name=email.split("@")[0])


def test_ingest_sets_owner_and_isolation(service) -> None:
    svc, _, users = service
    alice = _make_user(users, "alice@example.com")
    bob = _make_user(users, "bob@example.com")

    result = svc.ingest(alice.user_id, "doc.txt", b"private content for alice.")
    assert result.ingestion.status == "indexed"
    doc_id = result.ingestion.document_id

    owned = svc.get(alice.user_id, doc_id)
    assert owned.permission == SharingPermission.EDITOR
    assert owned.document.owner_id == alice.user_id

    assert [a.document_id for a in svc.list(bob.user_id)] == []
    with pytest.raises(DocumentNotFoundError):
        svc.get(bob.user_id, doc_id)
    with pytest.raises(DocumentNotFoundError):
        svc.delete(bob.user_id, doc_id)


def test_viewer_vs_editor_permissions(service) -> None:
    svc, _, users = service
    alice = _make_user(users, "alice@example.com")
    bob = _make_user(users, "bob@example.com")

    doc_id = svc.ingest(alice.user_id, "doc.txt", b"alpha content").ingestion.document_id

    svc.share(alice.user_id, doc_id, "bob@example.com", SharingPermission.VIEWER)
    shared = svc.get(bob.user_id, doc_id)
    assert shared.permission == SharingPermission.VIEWER
    assert shared.shared is True

    with pytest.raises(PermissionDeniedError):
        svc.update(bob.user_id, doc_id, "doc.txt", b"attempted edit")
    with pytest.raises(PermissionDeniedError):
        svc.delete(bob.user_id, doc_id)

    svc.share(alice.user_id, doc_id, "bob@example.com", SharingPermission.EDITOR)
    outcome = svc.update(bob.user_id, doc_id, "doc.txt", b"beta content from editor.")
    assert outcome.ingestion.status == "indexed"
    assert svc.get(bob.user_id, doc_id).permission == SharingPermission.EDITOR

    with pytest.raises(PermissionDeniedError):
        svc.delete(bob.user_id, doc_id)
    svc.delete(alice.user_id, doc_id)
    with pytest.raises(DocumentNotFoundError):
        svc.get(alice.user_id, doc_id)


def test_public_document_is_readonly_for_everyone(service) -> None:
    svc, repo, users = service
    alice = _make_user(users, "alice@example.com")
    bob = _make_user(users, "bob@example.com")

    from datetime import datetime, timezone

    public_doc = DocumentRecord(
        document_id="public_doc",
        filename="public.txt",
        content_hash="public-hash",
        page_count=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        owner_id=None,
    )
    repo.upsert_document(public_doc)

    for user in (alice, bob):
        access = svc.get(user.user_id, "public_doc")
        assert access.permission == SharingPermission.VIEWER
        assert access.document.owner_id is None

    with pytest.raises(PermissionDeniedError):
        svc.share(bob.user_id, "public_doc", "alice@example.com", SharingPermission.VIEWER)
    with pytest.raises(PermissionDeniedError):
        svc.update(bob.user_id, "public_doc", "public.txt", b"hijack")
    with pytest.raises(PermissionDeniedError):
        svc.delete(bob.user_id, "public_doc")


def test_share_requires_registered_recipient(service) -> None:
    svc, _, users = service
    alice = _make_user(users, "alice@example.com")
    doc_id = svc.ingest(alice.user_id, "doc.txt", b"content").ingestion.document_id

    from app.domain.exceptions import UserNotFoundError

    with pytest.raises(UserNotFoundError):
        svc.share(alice.user_id, doc_id, "ghost@example.com", SharingPermission.VIEWER)


def test_share_sets_upgrade_and_unshare(service) -> None:
    svc, _, users = service
    alice = _make_user(users, "alice@example.com")
    bob = _make_user(users, "bob@example.com")
    doc_id = svc.ingest(alice.user_id, "doc.txt", b"content").ingestion.document_id

    svc.share(alice.user_id, doc_id, "bob@example.com", SharingPermission.VIEWER)
    svc.share(alice.user_id, doc_id, "bob@example.com", SharingPermission.EDITOR)
    assert svc.get(bob.user_id, doc_id).permission == SharingPermission.EDITOR

    shares = {s.shared_with_email: s.permission for s in svc.list_shares(alice.user_id, doc_id)}
    assert shares == {"bob@example.com": SharingPermission.EDITOR}

    svc.unshare(alice.user_id, doc_id, "bob@example.com")
    with pytest.raises(DocumentNotFoundError):
        svc.get(bob.user_id, doc_id)