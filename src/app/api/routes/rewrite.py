"""Standalone rewrite endpoint — orchestrators inspect variants before deciding
whether/how to retrieve. Performs no retrieval and reads no document content.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.auth import enforce_rate_limit, get_current_user
from app.api.schemas import RewriteRequest, RewriteResponse
from app.domain.entities import User

router = APIRouter()


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite_endpoint(
    body: RewriteRequest,
    request: Request,
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> RewriteResponse:
    container = request.app.state.container
    result = container.rewrite_service.rewrite(
        query=body.query,
        strategy=body.strategy.value if body.strategy is not None else None,
        max_variants=body.max_variants,
    )
    return RewriteResponse(
        original=result.original,
        variants=result.variants,
        strategy=result.strategy,
    )