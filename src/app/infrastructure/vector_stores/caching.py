"""Transparent caching wrapper over the ``VectorStore`` protocol.

``HybridRetriever`` and everything above it keep talking to a ``VectorStore``;
this wrapper sits between them and the real store (Chroma / PgVector) and
caches ``search`` results.

Correctness properties:

- **No cross-scope leakage**: the cache key folds a canonical fingerprint of
  ``SearchFilters`` (the ``document_ids`` access-control scope, ``None``/``[]``
  distinguished) so a query served for one scope can never be replayed for
  another.
- **No stale results after ingestion**: ``IndexPersister`` bumps a per-collection
  generation counter on every write; the current value is part of the key, so a
  write makes every previously-served hit unreachable.
- **Stampede-safe**: on a miss, a ``SingleFlight`` primitive lets exactly one
  caller compute; the rest wait (bounded) or fall back to computing.
- **Self-healing**: any cache backend error degrades to a direct store call.

Cached chunks are re-validated and copied per request so a caller never
observes another request's ``query_id`` nor mutates shared objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from app.common.cache import Cache
from app.common.cache.generation import IndexGeneration
from app.common.cache.single_flight import SingleFlight
from app.domain.entities import EmbeddedChunk, RetrievedChunk
from app.domain.interfaces import VectorStore
from app.domain.value_objects import SearchFilters, VectorQuery

logger = logging.getLogger(__name__)

_KEY_SCHEMA = "vector-cache-v1"
_SENTINEL_NONE = "U"  # document_ids is None ⇒ "unfiltered" scope
_SENTINEL_EMPTY = "E"  # document_ids == [] ⇒ matches nothing (never == unfiltered)

_POLL_INTERVAL_SECONDS = 0.005


def build_vector_cache_key(query: VectorQuery, generation: int, schema_version: str = _KEY_SCHEMA) -> str:
    """Deterministic cache key for a vector search.

    Inputs (all but per-request ``query_id``): the query embedding (the actual
    ANN input — deterministic for a fixed model+version), the embedding model
    and version, ``top_k``, the canonicalized filters fingerprint, and the
    index generation.
    """
    digest = hashlib.sha256(json.dumps(query.query_embedding, separators=(",", ":")).encode("utf-8")).hexdigest()
    canonical = "\x00".join(
        [
            schema_version,
            query.embedding_model,
            str(query.embedding_version),
            str(query.top_k),
            _fingerprint(query.filters),
            digest,
        ]
    )
    inner = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"vcs:{generation}:{inner}"


def _fingerprint(filters: SearchFilters) -> str:
    scopes = []
    if filters.document_ids is None:
        scopes.append(_SENTINEL_NONE)
    elif not filters.document_ids:
        scopes.append(_SENTINEL_EMPTY)
    else:
        scopes.append("D:" + ",".join(sorted(filters.document_ids)))
    exact = json.dumps(filters.exact, sort_keys=True, separators=(",", ":"))
    must_not = json.dumps(filters.must_not, sort_keys=True, separators=(",", ":"))
    limit = f"L{filters.limit_per_document}" if filters.limit_per_document is not None else ""
    return "\x00".join(scopes + [exact, must_not, limit])


class CachingVectorStore:
    """``VectorStore`` wrapper that caches ``search`` results."""

    def __init__(
        self,
        store: VectorStore,
        cache: Cache,
        generation: IndexGeneration,
        single_flight: SingleFlight | None = None,
        ttl_seconds: int = 300,
        stampede_wait_seconds: float = 2.0,
    ) -> None:
        self._store = store
        self._cache = cache
        self._generation = generation
        self._single_flight = single_flight
        self._ttl = ttl_seconds
        self._stampede_wait = stampede_wait_seconds

        self.hits = 0
        self.misses = 0
        self.saved_ms = 0.0
        self.errors = 0

    # --- write path (delegated) ---

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self._store.upsert(chunks)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self._store.delete(chunk_ids)

    def get_count(self) -> int:
        return self._store.get_count()

    # --- read path (cached) ---

    def search(self, query: VectorQuery) -> list[RetrievedChunk]:
        key = build_vector_cache_key(query, self._generation.value())

        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            self.saved_ms += float(cached.get("compute_ms", 0.0))
            logger.debug(
                "vector cache hit; key=%s query_id=%s saved_ms=%.1f",
                key,
                query.query_id,
                cached.get("compute_ms", 0.0),
            )
            return self._decode(cached, query.query_id)

        self.misses += 1
        leader = self._single_flight is None or self._single_flight.try_acquire(key)
        if leader:
            try:
                cached = self._get(key)
                if cached is not None:
                    return self._decode(cached, query.query_id)
                return self._compute_and_store(query, key)
            finally:
                if self._single_flight is not None:
                    self._single_flight.release(key)

        return self._wait_for_result(query, key)

    def _wait_for_result(self, query: VectorQuery, key: str) -> list[RetrievedChunk]:
        deadline = time.monotonic() + self._stampede_wait
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)
            cached = self._get(key)
            if cached is not None:
                return self._decode(cached, query.query_id)
        cached = self._get(key)
        if cached is not None:
            return self._decode(cached, query.query_id)
        # Leader failed or timed out under the stampede window — compute
        # directly rather than fail the request.
        return self._compute_and_store(query, key)

    def _compute_and_store(self, query: VectorQuery, key: str) -> list[RetrievedChunk]:
        started = time.perf_counter()
        results = self._store.search(query)
        compute_ms = (time.perf_counter() - started) * 1000.0
        envelope = {
            "results": [c.model_dump(mode="json") for c in results],
            "compute_ms": round(compute_ms, 3),
        }
        self._set(key, json.dumps(envelope))
        return results

    # --- cache boundary (all cache failures degrade to uncached) ---

    def _get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._cache.get(key)
        except Exception as exc:  # noqa: BLE001 - cache must never break retrieval
            self.errors += 1
            logger.warning("vector cache get failed; degrading to uncached: %s", exc)
            return None
        if raw is None:
            return None
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def _set(self, key: str, payload: str) -> None:
        try:
            self._cache.set(key, payload, ttl_seconds=self._ttl)
        except Exception as exc:  # noqa: BLE001 - cache must never break retrieval
            self.errors += 1
            logger.warning("vector cache set failed; continuing uncached: %s", exc)

    @staticmethod
    def _decode(envelope: dict[str, Any], query_id: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk.model_validate(d).model_copy(update={"query_id": query_id})
            for d in envelope.get("results", [])
        ]

    # --- observability ---

    def cache_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": getattr(self._cache, "backend", "unknown"),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hits / total, 4) if total else 0.0,
            "saved_ms": round(self.saved_ms, 1),
            "errors": self.errors,
        }

    def info(self) -> dict[str, Any]:
        base = self._store.info()
        base["cache"] = self.cache_stats()
        return base


__all__ = ["CachingVectorStore", "build_vector_cache_key"]