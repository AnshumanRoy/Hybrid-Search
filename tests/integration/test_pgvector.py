"""PgVectorStore integration tests against a real PostgreSQL + pgvector.

Skipped unless a test PostgreSQL is available. Point one at it with:

    $env:RAG_TEST_POSTGRES_URL="postgresql+psycopg://rag:rag@localhost:5432/rag"
    python -m pytest tests/integration/test_pgvector.py -q

Start the bundled service with: docker compose up -d postgres
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from app.domain.exceptions import RetrievalError, VectorStoreError
from app.domain.value_objects import SearchFilters, VectorQuery
from app.infrastructure.vector_stores.pgvector import PgVectorStore
from tests.fakes import FakeEmbedder, make_chunk, make_embedded_chunk

TEST_DATABASE_URL = os.environ.get("RAG_TEST_POSTGRES_URL")


pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="set RAG_TEST_POSTGRES_URL to a PostgreSQL+pgvector database to run"
)


@pytest.fixture
def store() -> PgVectorStore:
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    table = f"test_vectors_{uuid.uuid4().hex[:12]}"
    store = PgVectorStore(engine=engine, table_name=table, dimension=8, embedding_model="fake-model")
    yield store
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
        conn.execute(text("DELETE FROM pgvector_store_meta WHERE table_name = :t"), {"t": table})
    engine.dispose()


def _query(query_id: str, embedding: list[float], top_k: int = 5, filters: SearchFilters | None = None) -> VectorQuery:
    return VectorQuery(
        query_id=query_id,
        query="",
        query_embedding=embedding,
        embedding_model="fake-model",
        embedding_version=1,
        top_k=top_k,
        filters=filters or SearchFilters(),
    )


def test_upsert_search_delete_roundtrip(store):
    embedder = FakeEmbedder()
    chunks = [
        make_embedded_chunk(make_chunk("doc_1", text="vacation policy", index=0), embedder),
        make_embedded_chunk(make_chunk("doc_1", text="refund window", index=1), embedder),
        make_embedded_chunk(make_chunk("doc_2", text="hr onboarding", index=0), embedder),
    ]
    store.upsert(chunks)
    assert store.get_count() == 3

    q = _query("q1", embedder.embed(["refund window"])[0], top_k=3)
    hits = store.search(q)
    assert hits[0].chunk_id == chunks[1].chunk.chunk_id
    assert {h.chunk_id for h in hits} == {c.chunk.chunk_id for c in chunks}
    assert len(hits) == 3
    assert hits[0].semantic_score is not None and 0.0 <= hits[0].semantic_score <= 1.0
    assert hits[0].filename == "doc_1.txt"
    assert hits[0].document_id == "doc_1"
    assert hits[0].metadata["document_type"] == "policy"

    store.delete([chunks[0].chunk.chunk_id])
    assert store.get_count() == 2

    info = store.info()
    assert info["name"].startswith("test_vectors_")
    assert info["backend"] == "pgvector"


def test_upsert_is_idempotent(store):
    embedder = FakeEmbedder()
    chunk = make_embedded_chunk(make_chunk("doc_1", text="same text", index=0), embedder)
    store.upsert([chunk])
    store.upsert([chunk])
    assert store.get_count() == 1


def test_filters_exact_and_must_not(store):
    embedder = FakeEmbedder()
    a = make_embedded_chunk(make_chunk("doc_1", text="finance policy", index=0, metadata={"department": "finance"}), embedder)
    b = make_embedded_chunk(make_chunk("doc_2", text="hr policy", index=0, metadata={"department": "hr"}), embedder)
    store.upsert([a, b])

    hits = store.search(_query("q1", embedder.embed(["policy"])[0], filters=SearchFilters(exact={"department": "finance"})))
    assert [h.chunk_id for h in hits] == [a.chunk.chunk_id]

    hits = store.search(_query("q2", embedder.embed(["policy"])[0], filters=SearchFilters(must_not={"department": "hr"})))
    assert [h.chunk_id for h in hits] == [a.chunk.chunk_id]


def test_model_mismatch_is_rejected(store):
    embedder = FakeEmbedder()
    chunk = make_embedded_chunk(make_chunk("doc_1", text="text", index=0), embedder)
    store.upsert([chunk])

    from app.domain.entities import EmbeddedChunk

    foreign = EmbeddedChunk(
        chunk=chunk.chunk,
        embedding=chunk.embedding,
        embedding_model="other-model",
        embedding_dimension=8,
        embedding_version=1,
    )
    with pytest.raises(VectorStoreError, match="refusing to upsert"):
        store.upsert([foreign])

    foreign_query = _query("q1", chunk.embedding)
    foreign_query = VectorQuery(
        query_id="q1",
        query="",
        query_embedding=chunk.embedding,
        embedding_model="other-model",
        embedding_version=1,
        top_k=5,
    )
    with pytest.raises(VectorStoreError, match="does not match store model"):
        store.search(foreign_query)


def test_dimension_mismatch_is_rejected(store):
    embedder = FakeEmbedder()
    chunk = make_embedded_chunk(make_chunk("doc_1", text="text", index=0), embedder)

    from app.domain.entities import EmbeddedChunk

    wrong_dim = EmbeddedChunk(
        chunk=chunk.chunk,
        embedding=[0.1, 0.2],
        embedding_model="fake-model",
        embedding_dimension=2,
        embedding_version=1,
    )
    with pytest.raises(VectorStoreError, match="dimension mismatch"):
        store.upsert([wrong_dim])

    with pytest.raises(RetrievalError, match="dims"):
        store.search(_query("q1", [0.1, 0.2]))