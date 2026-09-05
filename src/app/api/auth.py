"""FastAPI auth dependencies (bearer token → current user)."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from app.domain.entities import User
from app.domain.exceptions import AuthenticationError


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