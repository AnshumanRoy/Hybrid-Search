"""PostgreSQL/pgvector vector store.

Satisfies the ``VectorStore`` protocol using a dedicated ``chunk_vectors`` table
(tablename derived from the embedding model) with a ``vector`` column and an HNSW
cosine index. Vectors and metadata live in the same PostgreSQL instance that
backs the document store — one database, one operational surface.

Embedding versioning: the table name encodes the embedding model; a side-car
meta table records the exact model+dimension that created the table so vectors
produced by a different model can never be searched or mixed in silently.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql import cast

from app.common.configuration.settings import Settings
from app.domain.entities import EmbeddedChunk, RetrievedChunk
from app.domain.exceptions import RetrievalError, VectorStoreError
from app.domain.value_objects import SearchFilters, VectorQuery

logger = logging.getLogger(__name__)

_META_TABLE = "pgvector_store_meta"
_IDX_PREFIX = "ix"

ALLOWED_TABLE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


class PgVectorStore:
    """pgvector-backed persistent vector store satisfying the ``VectorStore`` protocol."""

    def __init__(
        self,
        engine: Engine,
        table_name: str,
        dimension: int,
        embedding_model: str,
    ) -> None:
        if not engine.dialect.name == "postgresql":
            raise VectorStoreError(
                f"PgVectorStore requires a PostgreSQL engine, got '{engine.dialect.name}'"
            )
        self._engine = engine
        self._table_name = self._sanitize(table_name)
        self._dimension = dimension
        self._embedding_model = embedding_model

        try:
            from pgvector.sqlalchemy import Vector
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("pgvector is not installed (pip install pgvector psycopg[binary])") from exc
        self._vector_type = Vector(self._dimension)

        self._table = Table(
            self._table_name,
            MetaData(),
            Column("chunk_id", String(96), primary_key=True),
            Column("document_id", String(64), nullable=False),
            Column("filename", String(512), nullable=False),
            Column("source_uri", String(2048), nullable=True),
            Column("page_number", Integer, nullable=True),
            Column("text", Text, nullable=False),
            Column("metadata_json", JSONB, nullable=False),
            Column("embedding_model", String(256), nullable=False),
            Column("embedding_version", Integer, nullable=False),
            Column("embedding", self._vector_type, nullable=False),
        )
        self._init_schema()

    @classmethod
    def from_settings(cls, settings: Settings, engine: Engine) -> "PgVectorStore":
        return cls(
            engine=engine,
            table_name=settings.vector_table_name,
            dimension=settings.vector_dimension,
            embedding_model=settings.embedding_model,
        )

    # ------------------------------------------------------------------ setup
    def _init_schema(self) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS "vector"'))
                self._table.create(conn, checkfirst=True)
                index_name = f"{_IDX_PREFIX}_{self._table_name}_embedding"
                conn.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                        f'ON "{self._table_name}" USING hnsw (embedding vector_cosine_ops)'
                    )
                )
                conn.execute(
                    text(
                        f'CREATE TABLE IF NOT EXISTS "{_META_TABLE}" ('
                        "table_name VARCHAR(96) PRIMARY KEY, "
                        "embedding_model VARCHAR(256) NOT NULL, "
                        "dimension INT NOT NULL)"
                    )
                )
                row = conn.execute(
                    text(f"SELECT embedding_model, dimension FROM {_META_TABLE} WHERE table_name = :t"),
                    {"t": self._table_name},
                ).fetchone()
                if row is not None and (row.embedding_model != self._embedding_model or row.dimension != self._dimension):
                    raise VectorStoreError(
                        f"vector table '{self._table_name}' already exists for embedding model "
                        f"'{row.embedding_model}' (dim={row.dimension}); refusing to mix incompatible vectors"
                    )
                conn.execute(
                    text(
                        f'INSERT INTO {_META_TABLE} (table_name, embedding_model, dimension) '
                        "VALUES (:t, :m, :d) ON CONFLICT (table_name) DO UPDATE SET "
                        "embedding_model = EXCLUDED.embedding_model, dimension = EXCLUDED.dimension"
                    ),
                    {"t": self._table_name, "m": self._embedding_model, "d": self._dimension},
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"failed to initialize pgvector schema: {exc}") from exc

    @staticmethod
    def _sanitize(name: str) -> str:
        cleaned = "".join(c for c in name.lower() if c in ALLOWED_TABLE_CHARS)
        return cleaned or "hybrid_search"

    # ------------------------------------------------------------------ write
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        model = chunks[0].embedding_model
        if model != self._embedding_model:
            raise VectorStoreError(
                f"refusing to upsert chunks embedded with '{model}' into a store configured for "
                f"'{self._embedding_model}'"
            )
        self._assert_no_foreign_model(model)

        rows: list[dict[str, Any]] = []
        for ec in chunks:
            if len(ec.embedding) != self._dimension:
                raise VectorStoreError(
                    f"embedding dimension mismatch: chunk '{ec.chunk.chunk_id}' has "
                    f"{len(ec.embedding)} dims, expected {self._dimension}"
                )
            rows.append(
                {
                    "chunk_id": ec.chunk.chunk_id,
                    "document_id": ec.chunk.document_id,
                    "filename": ec.chunk.filename,
                    "source_uri": ec.chunk.source_uri,
                    "page_number": ec.chunk.page_number,
                    "text": ec.chunk.text,
                    "metadata_json": ec.chunk.metadata,
                    "embedding_model": ec.embedding_model,
                    "embedding_version": ec.embedding_version,
                    "embedding": ec.embedding,
                }
            )

        insert = pg_insert(self._table).values(rows)
        on_conflict = insert.on_conflict_do_update(
            index_elements=[self._table.c.chunk_id],
            set_={
                "document_id": insert.excluded.document_id,
                "filename": insert.excluded.filename,
                "source_uri": insert.excluded.source_uri,
                "page_number": insert.excluded.page_number,
                "text": insert.excluded.text,
                "metadata_json": insert.excluded.metadata_json,
                "embedding_model": insert.excluded.embedding_model,
                "embedding_version": insert.excluded.embedding_version,
                "embedding": insert.excluded.embedding,
            },
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(on_conflict)
        except Exception as exc:
            raise VectorStoreError(f"pgvector upsert failed for {len(rows)} chunks: {exc}") from exc

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self._table.delete().where(self._table.c.chunk_id.in_(list(chunk_ids)))
                )
        except Exception as exc:
            raise VectorStoreError(f"pgvector delete failed: {exc}") from exc

    # ------------------------------------------------------------------ read
    def search(self, query: VectorQuery) -> list[RetrievedChunk]:
        if query.embedding_model != self._embedding_model:
            raise VectorStoreError(
                f"query embedding model '{query.embedding_model}' does not match store model "
                f"'{self._embedding_model}'; re-index with the matching embedder"
            )
        if len(query.query_embedding) != self._dimension:
            raise RetrievalError(
                f"query embedding has {len(query.query_embedding)} dims but store expects {self._dimension}"
            )

        q = cast(list(query.query_embedding), self._vector_type)
        distance = self._table.c.embedding.cosine_distance(q)
        stmt = (
            self._table.select()
            .add_columns((1.0 - distance).label("similarity"))
            .where(self._table.c.embedding_model == query.embedding_model)
            .order_by(distance)
            .limit(query.top_k)
        )
        stmt = self._apply_filters(stmt, query.filters)

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt)
                return [
                    RetrievedChunk(
                        chunk_id=r.chunk_id,
                        document_id=r.document_id,
                        filename=r.filename,
                        source_uri=r.source_uri,
                        page_number=r.page_number,
                        text=r.text,
                        metadata=dict(r.metadata_json or {}),
                        semantic_score=_clamp(r.similarity),
                        retrieval_metadata={
                            "semantic": True,
                            "distance": 1.0 - (r.similarity if r.similarity is not None else 0.0),
                            "backend": "pgvector",
                        },
                        query_id=query.query_id,
                    )
                    for r in rows
                ]
        except Exception as exc:
            raise RetrievalError(f"pgvector search failed: {exc}") from exc

    def get_count(self) -> int:
        try:
            with self._engine.connect() as conn:
                return int(conn.execute(func.count().select().select_from(self._table)).scalar_one())
        except Exception as exc:
            raise VectorStoreError(f"pgvector count failed: {exc}") from exc

    def info(self) -> dict[str, Any]:
        return {
            "count": self.get_count(),
            "name": self._table_name,
            "embeddings_model": self._embedding_model,
            "index": f"hnsw (vector_cosine_ops)",
            "backend": "pgvector",
        }

    # ------------------------------------------------------------------ helpers
    def _assert_no_foreign_model(self, model: str) -> None:
        try:
            with self._engine.connect() as conn:
                present = conn.execute(
                    text(f"SELECT DISTINCT embedding_model FROM {self._table_name} WHERE embedding_model <> :m"),
                    {"m": model},
                ).scalars().all()
        except Exception as exc:
            raise VectorStoreError(f"pgvector model check failed: {exc}") from exc
        if present:
            raise VectorStoreError(
                f"vector table '{self._table_name}' already contains vectors from "
                f"'{present[0]}'; refusing to mix models"
            )

    def _apply_filters(self, stmt, filters: SearchFilters):
        if filters.document_ids is not None:
            stmt = stmt.where(self._table.c.document_id.in_(filters.document_ids))
        for key, value in filters.exact.items():
            stmt = stmt.where(self._table.c.metadata_json[key].astext == str(value))
        for key, value in filters.must_not.items():
            stmt = stmt.where(self._table.c.metadata_json[key].astext != str(value))
        return stmt


def _clamp(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))