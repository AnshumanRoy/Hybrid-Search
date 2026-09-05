"""Health endpoint (lightweight, no external dependencies)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse
from app.application.services.rag import RAGService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    container = request.app.state.container
    vector_info = container.vector_store.info()
    return HealthResponse(
        status="ok",
        vector_store=vector_info,
        services={
            "reranker": container.reranker.model_name if container.reranker else None,
            "llm": container.llm.model_name,
            "embedder": container.embedder.model_name,
        },
    )