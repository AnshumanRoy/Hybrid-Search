"""Document endpoints: CRUD, ownership-aware listing and sharing.

Every route (except none) requires a bearer token; access is resolved against
the document's owner/public/shared state by ``DocumentService``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.auth import get_current_user
from app.api.schemas import (
    DocumentResponse,
    IngestResponse,
    ShareRequest,
    ShareResponse,
    UpdateDocumentResponse,
)
from app.domain.entities import SharingPermission, User
from app.domain.exceptions import (
    DocumentNotFoundError,
    PermissionDeniedError,
    UserNotFoundError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_response(request: Request, user: User, doc_id: str) -> DocumentResponse:
    container = request.app.state.container
    access = container.document_service.get(user.user_id, doc_id)
    record = access.document
    owner_email = None
    if record.owner_id:
        owner = container.users.get_user(record.owner_id)
        owner_email = owner.email if owner else None
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        source_uri=record.source_uri,
        content_hash=record.content_hash,
        page_count=record.page_count,
        status=record.status,
        created_at=record.created_at.isoformat(),
        chunk_count=len(container.repository.get_chunks_by_document(record.document_id)),
        owner_id=record.owner_id,
        owner_email=owner_email,
        permission="owner" if record.owner_id == user.user_id else access.permission.value,
        shared=access.shared,
    )


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DocumentNotFoundError):
        return HTTPException(status_code=404, detail=str(exc) or "document not found")
    if isinstance(exc, PermissionDeniedError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, UserNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="internal server error")


@router.get("/", response_model=list[DocumentResponse])
def list_documents(request: Request, user: User = Depends(get_current_user)) -> list[DocumentResponse]:
    container = request.app.state.container
    accessible = container.document_service.list(user.user_id)
    return [
        DocumentResponse(
            document_id=a.document_id,
            filename=a.document.filename,
            source_uri=a.document.source_uri,
            content_hash=a.document.content_hash,
            page_count=a.document.page_count,
            status=a.document.status,
            created_at=a.document.created_at.isoformat(),
            chunk_count=len(container.repository.get_chunks_by_document(a.document_id)),
            owner_id=a.document.owner_id,
            owner_email=(owner.email if (owner := container.users.get_user(a.document.owner_id)) else None)
            if a.document.owner_id else None,
            permission="owner" if a.document.owner_id == user.user_id else a.permission.value,
            shared=a.shared,
        )
        for a in accessible
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    try:
        return _document_response(request, user, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "document not found") from exc


@router.post("/ingest", response_model=list[IngestResponse])
async def ingest_uploaded_file(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> list[IngestResponse]:
    """Ingest a single uploaded file, owned by the authenticated user."""
    container = request.app.state.container
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    outcome = container.document_service.ingest(user.user_id, file.filename, data)
    result = outcome.ingestion
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
def ingest_dir(request: Request, user: User = Depends(get_current_user)) -> list[IngestResponse]:
    """Scan the configured data directory and ingest all new/changed files."""
    container = request.app.state.container
    results = container.ingestion_service.process_dir(owner_id=user.user_id)
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


@router.put("/{document_id}", response_model=UpdateDocumentResponse)
async def update_document(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> UpdateDocumentResponse:
    """Re-index a document (owner or editor). Identity may change on re-index."""
    container = request.app.state.container
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    try:
        outcome = container.document_service.update(
            user.user_id,
            document_id,
            file.filename or "document",
            data,
        )
    except Exception as exc:  # noqa: BLE001 - map to HTTP with _map_error below
        raise _map_error(exc) from exc
    return _updated_response(container, user, outcome)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> None:
    container = request.app.state.container
    try:
        container.document_service.delete(user.user_id, document_id)
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc) from exc


@router.post("/{document_id}/share", response_model=ShareResponse)
def share_document(
    document_id: str,
    body: ShareRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> ShareResponse:
    container = request.app.state.container
    try:
        share = container.document_service.share(
            user.user_id,
            document_id,
            body.user_email,
            SharingPermission(body.permission),
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc) from exc
    return ShareResponse(
        document_id=share.document_id,
        user_email=share.shared_with_email,
        permission=share.permission.value,
        created_at=share.created_at.isoformat(),
    )


@router.get("/{document_id}/shares", response_model=list[ShareResponse])
def list_shares(
    document_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> list[ShareResponse]:
    container = request.app.state.container
    try:
        shares = container.document_service.list_shares(user.user_id, document_id)
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc) from exc
    return [
        ShareResponse(
            document_id=s.document_id,
            user_email=s.shared_with_email,
            permission=s.permission.value,
            created_at=s.created_at.isoformat(),
        )
        for s in shares
    ]


@router.delete("/{document_id}/share/{user_email}", status_code=204)
def unshare_document(
    document_id: str,
    user_email: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> None:
    container = request.app.state.container
    try:
        container.document_service.unshare(user.user_id, document_id, user_email)
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc) from exc


def _updated_response(container, user: User, outcome) -> UpdateDocumentResponse:
    record = outcome.access.document
    owner_email = container.users.get_user(record.owner_id).email if record.owner_id else None
    return UpdateDocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        source_uri=record.source_uri,
        content_hash=record.content_hash,
        page_count=record.page_count,
        status=record.status,
        created_at=record.created_at.isoformat(),
        chunk_count=len(container.repository.get_chunks_by_document(record.document_id)),
        owner_id=record.owner_id,
        owner_email=owner_email,
        permission="owner" if record.owner_id == user.user_id else outcome.access.permission.value,
        shared=outcome.access.shared,
        ingest_status=outcome.ingestion.status,
        error=outcome.ingestion.error,
        chunks_embedded=outcome.ingestion.chunks_embedded,
    )