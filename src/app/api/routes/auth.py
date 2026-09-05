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