"""FastAPI auth dependencies (bearer token → current user)."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from app.domain.entities import User
from app.domain.exceptions import AuthenticationError, RateLimitExceededError


def get_current_user(
    request: Request,
    authorization: str = Header(default=""),
) -> User:
    container = request.app.state.container
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="missing or malformed bearer token")
    try:
        return container.auth_service.verify_token(token.strip())
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def enforce_rate_limit(request: Request, user: User = Depends(get_current_user)) -> None:
    """Per-user cost guardrail for cost-bearing endpoints (429 on overflow).

    ``user`` is resolved via ``get_current_user``; FastAPI caches shared
    dependencies per request, so the token is verified exactly once.
    """
    container = request.app.state.container
    window_seconds = container.settings.rate_limit_window_seconds
    try:
        container.rate_limiter.consume(user.user_id)
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(window_seconds)},
        ) from exc