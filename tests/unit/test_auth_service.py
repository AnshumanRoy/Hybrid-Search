"""AuthService tests against a real in-memory users repository (no network)."""

from __future__ import annotations

import pytest

from app.application.services.auth_service import AuthService
from app.common.configuration.settings import Settings
from app.domain.exceptions import AuthenticationError, InvalidCredentialsError, UserAlreadyExistsError
from app.infrastructure.persistence.models import init_db, make_engine, session_factory_for
from app.infrastructure.persistence.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository


@pytest.fixture
def service():
    engine = make_engine("sqlite://", in_memory=True)
    init_db(engine)
    users = SQLAlchemyUserRepository(session_factory_for(engine))
    return AuthService(users, Settings(auth_secret_key="unit-test-secret"))


def test_register_issues_valid_token(service) -> None:
    user = service.register("alice@example.com", "s3cure-pass", "Alice")
    token = service.issue_token(user)
    verified = service.verify_token(token)
    assert verified is not None
    assert verified.user_id == user.user_id
    assert verified.email == "alice@example.com"


def test_register_normalizes_email(service) -> None:
    user = service.register("  BOB@Example.COM ", "s3cure-pass")
    assert user.email == "bob@example.com"


def test_register_rejects_duplicate(service) -> None:
    service.register("alice@example.com", "s3cure-pass")
    with pytest.raises(UserAlreadyExistsError):
        service.register("ALICE@example.com", "s3cure-pass")


def test_register_rejects_short_password(service) -> None:
    with pytest.raises(ValueError):
        service.register("alice@example.com", "short")


def test_authenticate_success(service) -> None:
    service.register("alice@example.com", "s3cure-pass")
    result = service.authenticate("alice@example.com", "s3cure-pass")
    assert result.user.email == "alice@example.com"
    assert result.token


def test_authenticate_wrong_password(service) -> None:
    service.register("alice@example.com", "s3cure-pass")
    with pytest.raises(InvalidCredentialsError):
        service.authenticate("alice@example.com", "wrong-password")


def test_authenticate_unknown_email(service) -> None:
    with pytest.raises(InvalidCredentialsError):
        service.authenticate("ghost@example.com", "s3cure-pass")


def test_verify_token_rejects_bad_token(service) -> None:
    with pytest.raises(AuthenticationError):
        service.verify_token("garbage.token")


def test_verify_token_rejects_removed_user(service) -> None:
    from app.domain.entities import User

    token = service.issue_token(User(user_id="nonexistent-id", email="ghost@example.com"))
    with pytest.raises(AuthenticationError):
        service.verify_token(token)


def test_verify_token_rejects_revoked_token(service) -> None:
    user = service.register("alice@example.com", "s3cure-pass")
    token = service.issue_token(user)
    assert service.verify_token(token) is not None
    service.revoke_all_tokens(user.user_id)
    with pytest.raises(AuthenticationError):
        service.verify_token(token)


def test_expired_token_survives_before_revoke_only(service) -> None:
    user = service.register("alice@example.com", "s3cure-pass")
    service.revoke_all_tokens(user.user_id)
    result = service.authenticate("alice@example.com", "s3cure-pass")
    assert service.verify_token(result.token) is not None


def test_revoke_missing_user_raises(service) -> None:
    with pytest.raises(AuthenticationError):
        service.revoke_all_tokens("no-such-user")


def test_refresh_issues_new_token_at_current_version(service) -> None:
    user = service.register("alice@example.com", "s3cure-pass")
    first = service.issue_token(user)
    refreshed = service.issue_token(user)
    assert first != refreshed
    assert service.verify_token(refreshed) is not None