"""API request/response DTOs (separate from domain models)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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