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
    """Persists and queries document + chunk records (SQL by default)."""

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


class LexicalRetriever(Protocol):
    """Lexical (BM25) retrieval over indexed chunk text."""

    def add_documents(self, chunks: Sequence[Chunk]) -> None: ...

    def remove_documents(self, chunk_ids: Sequence[str]) -> None: ...

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...


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