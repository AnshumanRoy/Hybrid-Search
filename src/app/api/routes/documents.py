"""Document ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.api.schemas import IngestResponse, DocumentResponse
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository

router = APIRouter(prefix="/documents")


@router.post("/ingest", response_model=list[IngestResponse])
async def ingest_uploaded_file(request: Request, file: UploadFile = File(...)):
    """Ingest a single uploaded file via the ingestion pipeline."""
    container = request.app.state.container
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    result = container.ingestion_service.process_bytes(file.filename, data)
    return [
        IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            status=result.status,
            chunks_embedded=result.chunks_embedded,
            error=result.error,
            duration_ms=result.duration_ms,
        )
    ]


@router.post("/ingest-dir", response_model=list[IngestResponse])
def ingest_dir(request: Request):
    """Scan the configured data directory and ingest all new/changed files."""
    container = request.app.state.container
    results = container.ingestion_service.process_dir()
    return [
        IngestResponse(
            document_id=r.document_id,
            filename=r.filename,
            status=r.status,
            chunks_embedded=r.chunks_embedded,
            error=r.error,
            duration_ms=r.duration_ms,
        )
        for r in results
    ]


@router.get("/", response_model=list[DocumentResponse])
def list_documents(request: Request):
    container = request.app.state.container
    docs = container.repository.list_documents()
    return [
        DocumentResponse(
            document_id=d.document_id,
            filename=d.filename,
            source_uri=d.source_uri,
            content_hash=d.content_hash,
            page_count=d.page_count,
            status=d.status,
            created_at=d.created_at.isoformat(),
            chunk_count=len(container.repository.get_chunks_by_document(d.document_id)),
        )
        for d in docs
    ]