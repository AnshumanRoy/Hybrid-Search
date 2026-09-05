"""Cache abstraction (optional layer).

What should be cached:
  - LLM responses (same query + same context ⇒ same answer when temperature=0)
  - optionally, retrieval results for identical queries

What should NOT be cached (or cached with extreme care):
  - user PII / sensitive query text → keys are hashed; values are the response
  - embeddings (respond to embedding version changes; cache keyed by model+version if added)

The abstraction is intentionally tiny so it can be backed by memory, Redis,
or memcached without touching any consumer.
"""

from __future__ import annotations

import time
from typing import Any, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear(self) -> None: ...


class InMemoryCache:
    """Thread-safe TTL cache with a bounded entry count (LRU-ish eviction)."""

    def __init__(self, max_entries: int = 4096, default_ttl_seconds: int = 3600) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = __import__("threading").Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at < time.monotonic():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        expires_at = (time.monotonic() + ttl) if ttl and ttl > 0 else None
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                self._evict_one()
            self._data[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _evict_one(self) -> None:
        if not self._data:
            return
        oldest_key = min(self._data, key=lambda k: self._data[k][0] or float("inf"))
        del self._data[oldest_key]


__all__ = ["Cache", "InMemoryCache"]