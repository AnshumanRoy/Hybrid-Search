"""Query endpoint — POST /query returns grounded answer + structured citations."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.auth import enforce_rate_limit, get_current_user
from app.api.schemas import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)
from app.domain.entities import User
from app.domain.exceptions import LLMAuthenticationError, UnknownModelError

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    body: QueryRequest,
    request: Request,
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> QueryResponse:
    container = request.app.state.container
    qid = uuid.uuid4().hex

    try:
        response = container.rag_service.answer(
            query=body.query,
            top_k=body.top_k,
            query_id=qid,
            accessible_document_ids=container.document_service.accessible_document_ids(user.user_id),
            model=body.model,
            rewrite=body.rewrite,
            max_variants=body.max_variants,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMAuthenticationError as exc:
        # Provider rejected our key — surface as 401 (invalid credentials on
        # the configured backend), never as a generic 500.
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    citations = [
        CitationResponse(
            document=c.filename,
            page=c.page_number,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source_uri=c.source_uri,
        )
        for c in response.citations
    ]

    chunks = [
        RetrievedChunkResponse(
            chunk_id=rc.chunk_id,
            document_id=rc.document_id,
            filename=rc.filename,
            page_number=rc.page_number,
            text=rc.text,
            metadata=rc.metadata,
        )
        for rc in response.retrieved_chunks
    ]

    return QueryResponse(
        answer=response.answer,
        citations=citations,
        retrieved_chunks=chunks,
        query_id=response.metadata.get("query_id"),
        metadata=response.metadata,
    )