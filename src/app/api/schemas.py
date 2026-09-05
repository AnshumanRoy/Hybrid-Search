"""API request/response DTOs (separate from domain models)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.entities import SharingPermission


class IngestRequest(BaseModel):
    """Optional body for programmatic ingest of raw bytes."""

    filename: str = Field(min_length=1, max_length=255)
    data: bytes = Field(min_length=1, max_length=16 * 1024 * 1024)


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_embedded: int = 0
    error: str | None = None
    duration_ms: float | None = None


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    source_uri: str | None = None
    content_hash: str
    page_count: int | None = None
    status: str
    created_at: str
    chunk_count: int = 0
    owner_id: str | None = None
    owner_email: str | None = None
    permission: str = "viewer"
    shared: bool = False


class UpdateDocumentResponse(DocumentResponse):
    ingest_status: str | None = None
    error: str | None = None
    chunks_embedded: int = 0


class CitationResponse(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    document_id: str
    source_uri: str | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    query_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    vector_store: dict[str, Any] = Field(default_factory=dict)
    services: dict[str, Any] = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str | None = None
    created_at: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserResponse


class ShareRequest(BaseModel):
    user_email: str = Field(min_length=3, max_length=320)
    permission: SharingPermission = SharingPermission.VIEWER


class ShareResponse(BaseModel):
    document_id: str
    user_email: str
    permission: str
    created_at: str