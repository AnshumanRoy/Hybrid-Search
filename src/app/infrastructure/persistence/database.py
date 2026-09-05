"""Thin database wrapper owning engine + session factory.

Owns no business logic: repositories receive the session factory and translate
rows ↔ domain entities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.common.configuration.settings import Settings
from app.infrastructure.persistence.models import init_db


@dataclass
class Database:
    engine: Engine
    session_factory: sessionmaker

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        url = settings.sqlalchemy_url()
        if url.startswith("sqlite://"):
            connect_args = {"check_same_thread": False}
            if ":memory:" in url or url == "sqlite://":
                engine = create_engine("sqlite://", connect_args=connect_args, poolclass=StaticPool)
            else:
                db_path = Path(url.removeprefix("sqlite:///"))
                db_path.parent.mkdir(parents=True, exist_ok=True)
                engine = create_engine(url, connect_args=connect_args)
        else:
            engine = create_engine(url, pool_pre_ping=True)

        init_db(engine)
        return cls(engine=engine, session_factory=sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))

    def close(self) -> None:
        self.engine.dispose()