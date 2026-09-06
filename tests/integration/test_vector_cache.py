"""End-to-end vector-cache wiring through the real container (memory backend)."""

from __future__ import annotations

from app.bootstrap import build_container
from app.common.configuration.settings import Settings, VectorCacheBackend
from app.common.cache.generation import InMemoryIndexGeneration
from app.infrastructure.vector_stores.caching import CachingVectorStore
from app.domain.value_objects import SearchFilters
from tests.fakes import FakeEmbedder


def _container(tmp_path):
    settings = Settings(
        embedding_model="fake-cache-model",
        vector_dimension=8,
        embedding_batch_size=4,
        vector_store_dir=tmp_path / "vs",
        database_url="sqlite://",
        data_dir=tmp_path / "data",
        chunk_size=120,
        chunk_overlap=20,
        cache_enabled=False,
        vector_cache_backend=VectorCacheBackend.MEMORY,
        vector_cache_ttl_seconds=300,
    )
    return build_container(
        settings,
        with_reranker=False,
        embedder=FakeEmbedder(model_name="fake-cache-model"),
        vector_store=None,
    )


def test_container_wraps_store_and_serves_hits(tmp_path) -> None:
    container = _container(tmp_path)

    result = container.ingestion_service.process_bytes(
        "memo.txt", b"quarterly revenue grew three percent.", owner_id=None
    )
    doc_id = result.document_id
    assert result.status == "indexed"

    assert isinstance(container.vector_store, CachingVectorStore)
    assert container.vector_cache is not None
    assert isinstance(container.vector_cache_generation, InMemoryIndexGeneration)
    assert container.vector_cache_generation.value() == 1  # bumped by the ingest above

    filters = SearchFilters(document_ids=[doc_id])
    first = container.retrieval_service.retrieve("quarterly revenue", top_k=5, filters=filters)
    assert first, "expected at least one retrieved chunk"

    stats_after_miss = container.vector_store.cache_stats()
    assert (stats_after_miss["hits"], stats_after_miss["misses"]) == (0, 1)

    second = container.retrieval_service.retrieve("quarterly revenue", top_k=5, filters=filters)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    stats_after_hit = container.vector_store.cache_stats()
    assert stats_after_hit["hits"] == 1
    assert stats_after_hit["misses"] == 1


def test_ingest_invalidates_through_persister(tmp_path) -> None:
    container = _container(tmp_path)
    result = container.ingestion_service.process_bytes(
        "v1.txt", b"old version about taxes", owner_id=None
    )
    doc_id = result.document_id
    filters = SearchFilters(document_ids=[doc_id])
    query = "taxes"

    texts = lambda: [c.text for c in container.retrieval_service.retrieve(query, top_k=5, filters=filters)]
    assert any("old version" in t for t in texts())
    container.retrieval_service.retrieve(query, top_k=5, filters=filters)  # warm the cache

    container.ingestion_service.process_bytes("v1.txt", b"new version about taxes", owner_id=None)

    updated = [c.text for c in container.retrieval_service.retrieve(query, top_k=5, filters=filters)]
    assert any("new version" in t for t in updated)
    assert not any("old version" in t for t in updated)