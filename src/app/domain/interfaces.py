"""Domain interfaces (structural Protocols).

Application code depends on these protocols — never on concrete
infrastructure.  Concrete implementations live under ``app/infrastructure``
and are wired in at the composition root (``app/bootstrap.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from app.domain.entities import (
    Chunk,
    DocumentRecord,
    EmbeddedChunk,
    FileReference,
    ParsedDocument,
    RetrievedChunk,
    ShareRecord,
    SharingPermission,
    User,
)
from app.domain.value_objects import SearchFilters, VectorQuery


class DocumentParser(Protocol):
    """Extracts normalized, page-aware text from binary document source."""

    def parse(self, reference: FileReference, data: bytes) -> ParsedDocument: ...


class Chunker(Protocol):
    """Splits a parsed document into deterministic, meaningful chunks."""

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...


class Embedder(Protocol):
    """Computes normalized dense embeddings for a batch of texts.

    Must be deterministic for a fixed model + inputs,
    and must batch internally (never one text at a time).
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...


class VectorStore(Protocol):
    """Stores chunk embeddings and searches them via cosine similarity."""

    def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def search(self, query: VectorQuery) -> list[RetrievedChunk]: ...

    def get_count(self) -> int: ...

    def info(self) -> dict[str, Any]: ...


class DocumentRepository(Protocol):
    """Persists and queries document + chunk records (SQL by default).

    Access model: ``owner_id`` links a document to its creator (``None`` ⇒
    public/read-only); sharing grants viewers or editors read/update access.
    """

    def get_document(self, document_id: str) -> DocumentRecord | None: ...

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None: ...

    def list_documents(self) -> list[DocumentRecord]: ...

    def upsert_document(self, record: DocumentRecord) -> DocumentRecord: ...

    def delete_document(self, document_id: str) -> None: ...

    def add_chunks(self, chunks: Sequence[Chunk]) -> None: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]: ...

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

    def delete_chunks_by_document(self, document_id: str) -> None: ...

    # --- ownership & sharing ---
    def share_document(
        self,
        document_id: str,
        user_email: str,
        permission: SharingPermission,
    ) -> ShareRecord | None: ...

    def unshare_document(self, document_id: str, user_email: str) -> bool: ...

    def list_shares(self, document_id: str) -> list[ShareRecord]: ...

    def list_accessible_documents(self, user_id: str) -> list[DocumentRecord]: ...

    def list_accessible_document_ids(self, user_id: str) -> set[str]: ...

    def get_shared_permission(self, document_id: str, user_email: str) -> SharingPermission | None: ...


class UserRepository(Protocol):
    """Persists and queries user accounts."""

    def create_user(self, email: str, password_hash: str, display_name: str | None = None) -> User: ...

    def get_user(self, user_id: str) -> User | None: ...

    def get_user_by_email(self, email: str) -> User | None: ...

    def get_password_hash(self, email: str) -> str | None: ...

    def user_exists_with_email(self, email: str) -> bool: ...


class LexicalRetriever(Protocol):
    """Lexical (BM25) retrieval over indexed chunk text."""

    def add_documents(self, chunks: Sequence[Chunk]) -> None: ...

    def remove_documents(self, chunk_ids: Sequence[str]) -> None: ...

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...


class Persister(Protocol):
    """Receives fully-processed chunks and persists them (documents + chunks +
    embeddings + lexical index) as one transaction-like operation.
    """

    def find_by_hash(self, content_hash: str) -> DocumentRecord | None: ...

    def persist_document(
        self,
        document: ParsedDocument,
        embedded_chunks: list[EmbeddedChunk],
        status: str = "indexed",
        owner_id: str | None = None,
    ) -> None: ...

    def replace_document(
        self,
        document: ParsedDocument,
        embedded_chunks: list[EmbeddedChunk],
        owner_id: str | None = None,
    ) -> None: ...

    def delete_document(self, document_id: str) -> None: ...


class HybridSearch(Protocol):
    """Fuses semantic + lexical retrieval into one ranked candidate list."""

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        semantic_top_k: int | None = None,
        lexical_top_k: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    """Re-scores query–document pairs (order-preserving, metadata-preserving)."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]: ...


class LLM(Protocol):
    """Text-generation provider (local or remote)."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    @property
    def model_name(self) -> str: ...


class FileSource(Protocol):
    """Lists and reads files from an arbitrary source."""

    def list_files(self) -> Iterable[FileReference]: ...

    def read(self, file: FileReference) -> bytes: ...