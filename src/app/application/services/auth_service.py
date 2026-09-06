"""Application service for accounts, login and token issuance."""

from __future__ import annotations

import logging

from app.common.configuration.settings import Settings
from app.common.security import hash_password, sign_token, verify_password, verify_token
from app.domain.entities import User
from app.domain.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
)
from app.domain.interfaces import UserRepository

logger = logging.getLogger(__name__)

_MIN_PASSWORD_LENGTH = 8


class AuthResult:
    def __init__(self, user: User, token: str) -> None:
        self.user = user
        self.token = token


class AuthService:
    def __init__(self, users: UserRepository, settings: Settings) -> None:
        self.users = users
        self.settings = settings

    def register(self, email: str, password: str, display_name: str | None = None) -> User:
        email = email.strip().lower()
        if self.users.user_exists_with_email(email):
            raise UserAlreadyExistsError(f"a user with email {email!r} already exists")
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise ValueError(f"password must be at least {_MIN_PASSWORD_LENGTH} characters")
        user = self.users.create_user(email, hash_password(password), display_name=display_name)
        logger.info("registered user; user_id=%s email=%s", user.user_id, email)
        return user

    def authenticate(self, email: str, password: str) -> AuthResult:
        email = email.strip().lower()
        stored_hash = self.users.get_password_hash(email)
        if stored_hash is None or not verify_password(password, stored_hash):
            raise InvalidCredentialsError("invalid email or password")
        user = self.users.get_user_by_email(email)
        if user is None:  # pragma: no cover - hash necessarily implies a row
            raise InvalidCredentialsError("invalid email or password")
        return AuthResult(user=user, token=self.issue_token(user))

    def issue_token(self, user: User) -> str:
        return sign_token(
            {"sub": user.user_id, "email": user.email, "ver": user.token_version},
            self.settings.auth_secret_key,
            self.settings.auth_token_ttl_seconds,
        )

    def verify_token(self, token: str) -> User:
        payload = verify_token(token, self.settings.auth_secret_key)
        if payload is None:
            raise AuthenticationError("invalid or expired token")
        user = self.users.get_user(payload.get("sub", ""))
        if user is None:
            raise AuthenticationError("token references a removed user")
        if payload.get("ver", 0) != user.token_version:
            raise AuthenticationError("token revoked — please log in again")
        return user

    def revoke_all_tokens(self, user_id: str) -> None:
        """Invalidate every token ever issued to ``user_id``.

        Bumps ``users.token_version``; any token carrying an older ``ver``
        claim stops verifying immediately (stateless, no revocation list).
        """
        if self.users.increment_token_version(user_id) is None:
            raise AuthenticationError("no such user")
        logger.info("revoked all tokens; user_id=%s", user_id)