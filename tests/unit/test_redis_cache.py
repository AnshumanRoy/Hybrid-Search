"""RedisCache / RedisIndexGeneration / RedisSingleFlight tests.

Skipped unless a real Redis is available. Point one at it with:

    $env:RAG_TEST_REDIS_URL="redis://localhost:6379/0"
    python -m pytest tests/unit/test_redis_cache.py -q

Start the bundled service with: docker compose up -d redis
"""

from __future__ import annotations

import os

import pytest

from app.common.cache.generation import RedisIndexGeneration
from app.common.cache.redis import RedisCache
from app.common.cache.single_flight import RedisSingleFlight
from app.application.services.rewrite import RewriteService
from app.domain.value_objects import RewrittenQuery
from app.infrastructure.rewriters.heuristic import HeuristicQueryRewriter

TEST_REDIS_URL = os.environ.get("RAG_TEST_REDIS_URL")


pytestmark = pytest.mark.skipif(
    not TEST_REDIS_URL, reason="set RAG_TEST_REDIS_URL to a Redis instance to run"
)


@pytest.fixture
def client():
    redis = __import__("redis")
    c = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    c.flushdb()
    yield c
    c.flushdb()


def test_redis_cache_roundtrip_and_ttl(client) -> None:
    cache = RedisCache(client, prefix="vc:", ttl_seconds=300)
    assert cache.get("missing") is None
    cache.set("k", "hello", ttl_seconds=60)
    assert cache.get("k") == "hello"
    assert cache.get("k") is not None
    cache.set("other", {"a": [1, 2, 3]})
    assert cache.get("other") == {"a": [1, 2, 3]}
    cache.delete("k")
    assert cache.get("k") is None


def test_redis_cache_clear_scoped_to_prefix(client) -> None:
    a = RedisCache(client, prefix="vc:")
    b = RedisCache(client, prefix="other:")
    a.set("x", 1)
    b.set("y", 2)
    a.clear()
    assert a.get("x") is None
    assert b.get("y") == 2


def test_redis_index_generation_increments(client) -> None:
    gen = RedisIndexGeneration(client, key="vector_cache:index_generation:test")
    assert gen.value() == 0
    assert gen.bump() == 1
    assert gen.bump() == 2
    assert gen.value() == 2


def test_redis_single_flight_mutual_exclusion(client) -> None:
    sf = RedisSingleFlight(client, lease_seconds=1.0)
    key = "vcs:0:abc"
    assert sf.try_acquire(key) is True
    assert sf.try_acquire(key) is False  # already held (even by another instance)
    sf.release(key)
    assert sf.try_acquire(key) is True
    sf.release(key)


def test_redis_single_flight_lease_expiry(client) -> None:
    import time

    sf_short = RedisSingleFlight(client, lease_seconds=1.0)
    key = "vcs:0:lease"
    assert sf_short.try_acquire(key) is True
    time.sleep(1.1)
    other = RedisSingleFlight(client, lease_seconds=5.0)
    assert other.try_acquire(key) is True  # expired lease is re-acquirable
    other.release(key)


def test_rewrite_service_roundtrips_via_redis(client) -> None:
    from app.domain.interfaces import QueryRewriter

    class RecordingRewriter(QueryRewriter):
        def __init__(self) -> None:
            self.calls = 0

        def rewrite(
            self,
            query: str,
            *,
            max_variants: int,
            context_hint: str | None = None,
        ) -> RewrittenQuery:
            self.calls += 1
            return RewrittenQuery(original=query, variants=[query, f"{query}:v2"], strategy="heuristic")

    rewriter = RecordingRewriter()
    service = RewriteService(
        rewriter,
        HeuristicQueryRewriter(),
        cache=RedisCache(client, prefix="rewrite:", ttl_seconds=300),
    )
    first = service.rewrite("the refund policy", max_variants=3)
    assert first.variants == ["the refund policy", "the refund policy:v2"]
    second = service.rewrite("the refund policy", max_variants=3)
    assert second == first
    assert rewriter.calls == 1  # second call was a cache hit
    assert isinstance(second, RewrittenQuery)