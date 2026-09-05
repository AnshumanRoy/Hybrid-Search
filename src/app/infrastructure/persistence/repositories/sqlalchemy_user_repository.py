"""SQLAlchemy-backed user repository (implements the domain Protocol)."""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from app.domain.entities import User
from app.domain.interfaces import UserRepository
from app.infrastructure.persistence.models import UserRow


class SQLAlchemyUserRepository(UserRepository):
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def create_user(self, email: str, password_hash: str, display_name: str | None = None) -> User:
        with self._session_factory() as session:
            user = User(email=email, display_name=display_name)
            session.add(
                UserRow(
                    user_id=user.user_id,
                    email=email,
                    password_hash=password_hash,
                    display_name=display_name,
                    created_at=user.created_at,
                )
            )
            session.commit()
            return user

    def get_user(self, user_id: str) -> User | None:
        with self._session_factory() as session:
            row = session.get(UserRow, user_id)
            return self._to_user(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._session_factory() as session:
            row = session.query(UserRow).filter(UserRow.email == email.lower()).first()
            return self._to_user(row) if row else None

    def get_password_hash(self, email: str) -> str | None:
        with self._session_factory() as session:
            row = session.query(UserRow).filter(UserRow.email == email.lower()).first()
            return row.password_hash if row else None

    def user_exists_with_email(self, email: str) -> bool:
        with self._session_factory() as session:
            return session.query(UserRow).filter(UserRow.email == email.lower()).first() is not None

    @staticmethod
    def _to_user(row: UserRow) -> User:
        return User(
            user_id=row.user_id,
            email=row.email,
            display_name=row.display_name,
            created_at=row.created_at,
        )