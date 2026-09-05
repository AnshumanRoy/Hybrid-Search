"""SQL persistence.

SQLAlchemy lives entirely inside this package; repositories translate between
domain entities and ORM rows so no SQLAlchemy types leak upward.
"""

from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository

__all__ = ["Database", "SQLAlchemyDocumentRepository"]