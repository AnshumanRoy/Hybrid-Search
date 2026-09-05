"""Query endpoint — POST /query returns grounded answer + structured citations."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from app.api.schemas import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    container = request.app.state.container
    qid = uuid.uuid4().hex

    response = container.rag_service.answer(
        query=body.query,
        top_k=body.top_k,
        query_id=qid,
    )

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