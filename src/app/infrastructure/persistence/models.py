"""ORM models and schema initialization."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    source_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="indexed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChunkRow id={self.chunk_id} doc={self.document_id} page={self.page_number}>"


class QueryLogRow(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String(64), index=True)
    query_hash: Mapped[str] = mapped_column(String(64), index=True)
    response: Mapped[str] = mapped_column(Text)
    answerable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Content is intentionally NOT stored: we only keep a hash for duplicates.
    def __repr__(self) -> str:  # pragma: no cover
        return f"<QueryLogRow id={self.id} answerable={self.answerable}>"


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def session_factory_for(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def make_engine(database_url: str, *, in_memory: bool = False, echo: bool = False):
    if in_memory:
        return create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=echo,
        )
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, echo=echo, future=True)


__all__ = ["Base", "DocumentRow", "ChunkRow", "QueryLogRow", "init_db", "session_factory_for", "make_engine"]