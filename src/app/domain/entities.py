"""Domain entities (Pydantic models).

These are plain-data, versioned models. They carve out the stable language
shared between ingestion, retrieval and generation so that swapping
infrastructure never leaks into the domain.
"""

from __future__ import annotations

import enum
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SharingPermission(str, enum.Enum):
    """Permission level granted to a share recipient."""

    VIEWER = "viewer"
    EDITOR = "editor"


class User(BaseModel):
    """An authenticated principal who owns documents and shares them.

    ``token_version`` is a per-user revocation counter. It is embedded in every
    issued access token (``ver`` claim); bumping it invalidates every token ever
    issued to this user (``POST /auth/revoke``).
    """

    user_id: str = Field(default_factory=_new_id)
    email: str
    display_name: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    token_version: int = 0


class ShareRecord(BaseModel):
    """A document shared with another user (viewer or editor)."""

    document_id: str
    shared_with_email: str
    permission: SharingPermission
    created_at: datetime = Field(default_factory=utcnow)


class PageRange(BaseModel):
    start: int
    end: int


class FileReference(BaseModel):
    """Identifies a file/resource within a source.

    ``path`` is allowed to be any opaque string the source understands
    (local relative path, S3 key, blob path, ...).
    """

    source_id: str = "default"
    path: str
    filename: str | None = None
    uri: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.filename or self.path.split("/")[-1].split("\\")[-1]


class ParsedDocument(BaseModel):
    document_id: str = Field(default_factory=_new_id)
    filename: str
    source_uri: str | None = None
    source_id: str | None = None
    content_hash: str
    page_count: int
    pages: list["DocumentPage"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DocumentPage(BaseModel):
    document_id: str | None = None
    page_number: int = Field(ge=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def paragraph_blocks(self) -> list[str]:
        """Split a page into coarse paragraph blocks (blank-line separated)."""
        blocks: list[str] = []
        for block in self.text.split("\n\n"):
            block = " ".join(line.strip() for line in block.splitlines()).strip()
            if block:
                blocks.append(block)
        return blocks


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=_new_id)
    document_id: str
    filename: str
    source_uri: str | None = None
    page_number: int | None = None
    text: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int
    embedding_version: int


class RetrievedChunk(BaseModel):
    """A chunk returned from any retrieval path.

    ``retrieval_metadata`` holds stage-specific info (e.g. which retrievers
    contributed, pre/per-fusion scores, ...); ``query_id`` lets downstream
    monitoring group chunks belonging to one query.
    """

    chunk_id: str
    document_id: str
    filename: str
    source_uri: str | None = None
    page_number: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    semantic_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    fused_score: float | None = None
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    query_id: str | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        semantic_score: float | None = None,
        lexical_score: float | None = None,
        rerank_score: float | None = None,
        fused_score: float | None = None,
        retrieval_metadata: dict[str, Any] | None = None,
        query_id: str | None = None,
    ) -> "RetrievedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            source_uri=chunk.source_uri,
            page_number=chunk.page_number,
            text=chunk.text,
            metadata=chunk.metadata,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            rerank_score=rerank_score,
            fused_score=fused_score,
            retrieval_metadata=retrieval_metadata or {},
            query_id=query_id,
        )


class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: int | None = None
    chunk_id: str
    source_uri: str | None = None


class DocumentRecord(BaseModel):
    """Persisted record for a document (as stored in the SQL repository).

    ``owner_id`` is the user_id of the creator; ``None`` means the document is
    public (readable by any authenticated user, editable by nobody).
    """

    document_id: str
    filename: str
    source_uri: str | None = None
    source_id: str | None = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_count: int | None = None
    status: str = "indexed"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    owner_id: str | None = None


class ChunkRecord(BaseModel):
    """Persisted record for a chunk."""

    chunk_id: str
    document_id: str
    filename: str
    source_uri: str | None = None
    page_number: int | None = None
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRecord(BaseModel):
    """Diagnostic record of a RAG query."""

    query_id: str = Field(default_factory=_new_id)
    query: str
    response: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None
    retrieval_count: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def answers_question(self) -> bool:
        """Heuristic: whether the answer is a real answer, not an "unavailable" notice."""
        lowered = self.answer.lower()
        markers = (
            "cannot answer",
            "cannot determine",
            "not available",
            "unable to answer",
            "does not contain enough information",
            "insufficient",
            "no information",
            "not find",
            "could not find",
        )
        if any(m in lowered for m in markers):
            return False
        # Generic refusal transcriptions.
        if len(self.answer.split()) < 3:
            return False
        return True


class DocumentAccess(BaseModel):
    """A document together with the caller's effective permission."""

    document: DocumentRecord
    permission: SharingPermission
    shared: bool = False

    @property
    def document_id(self) -> str:
        return self.document.document_id


class EvaluationSample(BaseModel):
    question: str
    expected_answer: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)


class IngestionResult(BaseModel):
    document_id: str
    filename: str
    status: str  # "indexed" | "skipped" | "replaced" | "failed"
    chunks_embedded: int = 0
    error: str | None = None
    duration_ms: float | None = None


class EmbeddingInfo(BaseModel):
    embedding_model: str
    embedding_dimension: int
    embedding_version: int


# Keep the __all__ narrow to avoid unintended circulars.
__all__ = [
    "PageRange",
    "FileReference",
    "ParsedDocument",
    "DocumentPage",
    "Chunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "Citation",
    "DocumentRecord",
    "DocumentAccess",
    "ChunkRecord",
    "QueryRecord",
    "RAGResponse",
    "EvaluationSample",
    "IngestionResult",
    "EmbeddingInfo",
    "SharingPermission",
    "User",
    "ShareRecord",
    "utcnow",
]