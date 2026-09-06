"""Vector-store caching layer unit tests.

Covers: cache-key schema/scope isolation, hit/miss/eviction, invalidation on
generation bump, stampede protection under concurrency, graceful degradation
when the cache backend fails, and the IndexPersister generation hook.
"""

from __future__ import annotations

import threading
import time

from app.common.cache import InMemoryCache
from app.common.cache.generation import InMemoryIndexGeneration
from app.common.cache.single_flight import ThreadingSingleFlight
from app.domain.entities import DocumentPage, ParsedDocument
from app.domain.value_objects import SearchFilters, VectorQuery
from app.infrastructure.persistence.persister import IndexPersister
from app.infrastructure.vector_stores.caching import CachingVectorStore, build_vector_cache_key
from tests.fakes import FakeCache, FakeEmbedder, FakeVectorStore, make_chunk, make_embedded_chunk


# --------------------------------------------------------------------------- helpers


class ProbeVectorStore:
    """VectorStore double that counts ``search`` calls."""

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.searches = 0

    def search(self, query: VectorQuery):
        self.searches += 1
        return self.wrapped.search(query)

    def upsert(self, chunks) -> None:
        self.wrapped.upsert(chunks)

    def delete(self, chunk_ids) -> None:
        self.wrapped.delete(chunk_ids)

    def get_count(self) -> int:
        return self.wrapped.get_count()

    def info(self) -> dict:
        return self.wrapped.info()


class BlockingVectorStore(ProbeVectorStore):
    """Blocks the first search until ``release`` is set (stampede tests)."""

    def __init__(self, wrapped) -> None:
        super().__init__(wrapped)
        self.first_started = threading.Event()
        self.release = threading.Event()

    def search(self, query: VectorQuery):
        with threading.Lock():
            self.searches += 1
        if self.searches == 1:
            self.first_started.set()
            assert self.release.wait(timeout=10), "test harness failed to release leader"
        return self.wrapped.search(query)


def _query(
    text: str = "some query",
    query_id: str = "qid",
    filters: SearchFilters | None = None,
    top_k: int = 5,
    model: str = "fake-model",
    version: int = 1,
) -> VectorQuery:
    embedding = FakeEmbedder(model_name=model, version=version).embed([text])[0]
    return VectorQuery(
        query_id=query_id,
        query=text,
        query_embedding=embedding,
        embedding_model=model,
        embedding_version=version,
        top_k=top_k,
        filters=filters or SearchFilters(),
    )


def _seeded_store() -> FakeVectorStore:
    store = FakeVectorStore(dimension=8)
    embedder = FakeEmbedder()
    store.upsert(
        [
            make_embedded_chunk(make_chunk("doc_1", chunk_id="c1", text="alpha beta gamma", index=0), embedder),
            make_embedded_chunk(make_chunk("doc_1", chunk_id="c2", text="delta epsilon zeta", index=1), embedder),
        ]
    )
    return store


def _parsed_document(document_id: str = "doc_1") -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="f.txt",
        source_uri=None,
        source_id=None,
        content_hash="hash",
        page_count=1,
        pages=[DocumentPage(page_number=1, text="hello world")],
        metadata={},
    )


class StubRepository:
    def __init__(self) -> None:
        self.chunks: list = []

    def get_document_by_hash(self, content_hash):  # noqa: ARG002
        return None

    def upsert_document(self, record):
        return record

    def add_chunks(self, chunks) -> None:
        pass

    def get_chunks_by_document(self, document_id):
        return [c for c in self.chunks if c.document_id == document_id]

    def delete_chunks_by_document(self, document_id) -> None:
        self.chunks = [c for c in self.chunks if c.document_id != document_id]

    def delete_document(self, document_id) -> None:  # noqa: ARG002
        pass


class StubVector:
    def upsert(self, chunks) -> None:  # noqa: ARG002
        pass

    def delete(self, chunk_ids) -> None:  # noqa: ARG002
        pass


class StubLexical:
    def add_documents(self, chunks) -> None:  # noqa: ARG002
        pass

    def remove_documents(self, chunk_ids) -> None:  # noqa: ARG002
        pass


# --------------------------------------------------------------------------- cache key schema


def test_key_builder_separates_access_scopes() -> None:
    base = _query(filters=SearchFilters(document_ids=["a", "b"]))
    unfiltered = _query(filters=None)
    empty = _query(filters=SearchFilters(document_ids=[]))
    only_a = _query(filters=SearchFilters(document_ids=["a"]))

    keys = {
        build_vector_cache_key(base, 0),
        build_vector_cache_key(unfiltered, 0),
        build_vector_cache_key(empty, 0),
        build_vector_cache_key(only_a, 0),
    }
    assert len(keys) == 4  # None / [] / [a] / [a,b] are all distinct buckets


def test_key_builder_is_ordering_and_query_id_invariant() -> None:
    ab = _query(filters=SearchFilters(document_ids=["a", "b"]))
    ba = _query(filters=SearchFilters(document_ids=["b", "a"]))
    assert build_vector_cache_key(ab, 0) == build_vector_cache_key(ba, 0)

    q1 = _query(query_id="first-request")
    q2 = _query(query_id="second-request")
    assert build_vector_cache_key(q1, 0) == build_vector_cache_key(q2, 0)


def test_key_builder_sensitivity_to_model_version_top_k_generation() -> None:
    base = _query()
    assert build_vector_cache_key(base, 0) != build_vector_cache_key(_query(model="other-model"), 0)
    assert build_vector_cache_key(base, 0) != build_vector_cache_key(_query(version=2), 0)
    assert build_vector_cache_key(base, 0) != build_vector_cache_key(_query(top_k=3), 0)
    assert build_vector_cache_key(base, 0) != build_vector_cache_key(base, 1)


def test_key_changes_with_exact_and_limit_filters() -> None:
    exact = _query(filters=SearchFilters(exact={"department": "finance"}))
    plain = _query(filters=SearchFilters())
    limited = _query(filters=SearchFilters(limit_per_document=2))
    assert build_vector_cache_key(exact, 0) != build_vector_cache_key(plain, 0)
    assert build_vector_cache_key(limited, 0) != build_vector_cache_key(plain, 0)


# --------------------------------------------------------------------------- hit / miss / eviction


def test_cache_hit_after_miss_and_counters() -> None:
    underlying = ProbeVectorStore(_seeded_store())
    store = CachingVectorStore(underlying, FakeCache(), InMemoryIndexGeneration())

    q = _query(filters=SearchFilters(document_ids=["doc_1"]))
    first = store.search(q)
    assert [c.chunk_id for c in first] == ["c1", "c2"]
    assert underlying.searches == 1
    assert (store.hits, store.misses) == (0, 1)

    second = store.search(_query(filters=SearchFilters(document_ids=["doc_1"])))
    assert [c.chunk_id for c in second] == ["c1", "c2"]
    assert underlying.searches == 1  # second request served from cache
    assert (store.hits, store.misses) == (1, 1)
    assert store.cache_stats()["hit_ratio"] == 0.5
    assert store.info()["cache"]["hits"] == 1


def test_inmemory_cache_evicts_oldest_when_full() -> None:
    cache = InMemoryCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None  # evicted
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_cache_ttl_expiry() -> None:
    cache = FakeCache()
    cache.set("k", "v", ttl_seconds=1)
    assert cache.get("k") == "v"
    cache._data["k"] = (time.monotonic() - 1, "v")  # force expiry
    assert cache.get("k") is None
    assert len(cache) == 0


def test_cache_hit_reassigns_query_id() -> None:
    store = CachingVectorStore(ProbeVectorStore(_seeded_store()), FakeCache(), InMemoryIndexGeneration())
    store.search(_query(filters=SearchFilters(document_ids=["doc_1"]), query_id="first"))
    hit = store.search(_query(filters=SearchFilters(document_ids=["doc_1"]), query_id="second"))
    assert hit[0].query_id == "second"


# --------------------------------------------------------------------------- scope isolation


def test_scope_isolation_never_shares_cache_entries() -> None:
    underlying = ProbeVectorStore(_seeded_store())
    store = CachingVectorStore(underlying, FakeCache(), InMemoryIndexGeneration())

    q = _query(text="alpha beta", filters=SearchFilters(document_ids=["doc_1"]))
    q_other = _query(text="alpha beta", filters=SearchFilters(document_ids=["doc_x"]))
    q_unfiltered = _query(text="alpha beta", filters=None)
    q_empty = _query(text="alpha beta", filters=SearchFilters(document_ids=[]))

    assert store.search(q) != []
    assert store.search(q_other) == []  # doc_1 not in scope doc_x
    assert store.search(q_unfiltered) != []
    assert store.search(q_empty) == []  # empty scope must not see anything

    assert underlying.searches == 4  # none of the four scopes shared an entry

    # Re-running the same scope later is a hit within that scope only.
    store.search(q)
    assert underlying.searches == 4


# --------------------------------------------------------------------------- invalidation


def test_generation_bump_prevents_stale_results() -> None:
    embedder = FakeEmbedder()
    underlying = FakeVectorStore(dimension=8)
    gen = InMemoryIndexGeneration()
    store = CachingVectorStore(ProbeVectorStore(underlying), FakeCache(), gen)

    underlying.upsert(
        [make_embedded_chunk(make_chunk("doc_x", chunk_id="c_old", text="old version about taxes", index=0), embedder)]
    )
    q = _query(text="taxes", filters=SearchFilters(document_ids=["doc_x"]))

    assert [c.chunk_id for c in store.search(q)] == ["c_old"]

    # Ingest a change: new content replaces the old chunk.
    gen.bump()
    underlying.delete(["c_old"])
    underlying.upsert(
        [make_embedded_chunk(make_chunk("doc_x", chunk_id="c_new", text="new version about taxes", index=0), embedder)]
    )

    assert [c.chunk_id for c in store.search(q)] == ["c_new"]  # never the stale hit


def test_index_persister_bumps_generation_on_write_and_delete() -> None:
    gen = InMemoryIndexGeneration()
    persister = IndexPersister(StubRepository(), StubVector(), StubLexical(), generation=gen)

    assert gen.value() == 0
    persister.persist_document(_parsed_document("doc_1"), [])
    assert gen.value() == 1
    persister.persist_document(_parsed_document("doc_1"), [])
    assert gen.value() == 2
    persister.delete_document("doc_1")
    assert gen.value() == 3
    persister.replace_document(_parsed_document("doc_1"), [])
    assert gen.value() == 4


# --------------------------------------------------------------------------- stampede protection


def test_stampede_single_flight_computes_once_for_identical_requests() -> None:
    embedder = FakeEmbedder()
    underlying = BlockingVectorStore(_seeded_store())
    store = CachingVectorStore(
        underlying,
        FakeCache(),
        InMemoryIndexGeneration(),
        single_flight=ThreadingSingleFlight(),
        ttl_seconds=300,
        stampede_wait_seconds=5.0,
    )

    n_workers = 8
    barrier = threading.Barrier(n_workers)
    results: list[list] = []
    lock = threading.Lock()

    def worker(worker_id: str) -> None:
        barrier.wait()
        out = store.search(_query(text="alpha beta", filters=SearchFilters(document_ids=["doc_1"]), query_id=worker_id))
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(n_workers)]
    for t in threads:
        t.start()

    assert underlying.first_started.wait(timeout=5)
    time.sleep(0.3)  # let the other workers reach the polling path
    assert underlying.searches == 1
    underlying.release.set()

    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)

    assert underlying.searches == 1  # exactly one hit the underlying store
    assert len(results) == n_workers
    expected = sorted(["c1", "c2"])
    for out in results:
        assert sorted(c.chunk_id for c in out) == expected
    # each caller received its own query_id, never another request's
    assert {out[0].query_id for out in results} == {f"w{i}" for i in range(n_workers)}


# --------------------------------------------------------------------------- graceful degradation


def test_cache_backend_failure_degrades_to_uncached() -> None:
    underlying = ProbeVectorStore(_seeded_store())
    store = CachingVectorStore(
        underlying,
        FakeCache(fail=True),
        InMemoryIndexGeneration(),
        single_flight=ThreadingSingleFlight(),
    )

    out = store.search(_query(filters=SearchFilters(document_ids=["doc_1"])))
    assert [c.chunk_id for c in out] == ["c1", "c2"]  # still servable
    assert store.errors >= 1