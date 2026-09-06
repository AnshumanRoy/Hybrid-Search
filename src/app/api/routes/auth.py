"""Authentication endpoints: register, login, who-am-I."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.auth import get_current_user
from app.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.domain.entities import User
from app.domain.exceptions import InvalidCredentialsError, UserAlreadyExistsError

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at.isoformat(),
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, request: Request) -> TokenResponse:
    """Register a new account; returns an access token immediately."""
    container = request.app.state.container
    try:
        user = container.auth_service.register(body.email, body.password, body.display_name)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    token = container.auth_service.issue_token(user)
    return TokenResponse(token=token, user=_user_response(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    container = request.app.state.container
    try:
        result = container.auth_service.authenticate(body.email, body.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return TokenResponse(token=result.token, user=_user_response(result.user))


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return _user_response(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: Request, user: User = Depends(get_current_user)) -> TokenResponse:
    """Issues a fresh access token for an already-authenticated user.

    Rotates the access token without requiring re-login; the new token keeps
    the current ``token_version`` unless the user has since been revoked.
    """
    token = request.app.state.container.auth_service.issue_token(user)
    return TokenResponse(token=token, user=_user_response(user))


@router.post("/revoke", status_code=204)
def revoke(request: Request, user: User = Depends(get_current_user)) -> None:
    """Revokes every token ever issued to the current user.

    Bumps ``users.token_version``; every outstanding access token stops
    verifying immediately, so this doubles as "log out everywhere".
    """
    request.app.state.container.auth_service.revoke_all_tokens(user.user_id)