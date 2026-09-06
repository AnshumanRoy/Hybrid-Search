"""Per-collection index generation counter.

Used by the vector-store caching layer to keep cached hits consistent with
ingestion writes: ``IndexPersister`` bumps the generation on every write, and
the ``CachingVectorStore`` folds the current value into its cache keys. A write
therefore makes every previously-served hit for that collection unreachable
immediately — stale results can never be returned after a write.

Two backends:

- ``InMemoryIndexGeneration`` — process-local counter (dev / single replica).
- ``RedisIndexGeneration`` — atomic ``INCR`` on a Redis key, so invalidation is
  visible across all replicas sharing a cache.
"""

from __future__ import annotations

import threading
from typing import Protocol

GENERATION_PROTOCOL = "IndexGeneration"


class IndexGeneration(Protocol):
    """A monotonically increasing version counter for an index collection."""

    def value(self) -> int:
        """Current generation (0 ⇒ never written since this process started)."""
        ...

    def bump(self) -> int:
        """Increment and return the new generation.

        Called by the write path (``IndexPersister``) after every mutation.
        """
        ...


class InMemoryIndexGeneration:
    """Thread-safe process-local counter."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def value(self) -> int:
        with self._lock:
            return self._value

    def bump(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def backend(self) -> str:
        return "memory"


class RedisIndexGeneration:
    """Redis-backed counter shared across replicas.

    The key is created lazily on the first ``bump``; an absent key reads as 0.
    """

    def __init__(
        self,
        client: object,
        key: str = "vector_cache:index_generation",
    ) -> None:
        self._client = _as_redis(client)
        self._key = key
        self._client.ping()

    def value(self) -> int:
        raw = self._client.get(self._key)
        return int(raw) if raw is not None else 0

    def bump(self) -> int:
        return int(self._client.incr(self._key))

    @property
    def backend(self) -> str:
        return "redis"


def _as_redis(client: object) -> object:
    return client


__all__ = ["IndexGeneration", "InMemoryIndexGeneration", "RedisIndexGeneration"]