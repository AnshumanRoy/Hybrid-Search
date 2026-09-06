"""Request-coalescing (stampede protection) for the vector-store cache.

Under concurrent identical (query, filters) requests on a cache miss, exactly
one caller should compute; the rest wait for (or eventually share) that result.
The mechanism is deliberately backend-shaped:

- ``ThreadingSingleFlight`` — in-process per-key ``threading.Lock``.
- ``RedisSingleFlight`` — distributed ``SET NX PX`` lease; contenders poll the
  result key.

Both expose the same two primitives and are best-effort: if a leader dies (or a
Redis lease expires), a redundant compute may occur but a wrong answer can not.
"""

from __future__ import annotations

import secrets
import threading
from typing import Protocol

_ACQUIRE_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class SingleFlight(Protocol):
    """Non-blocking try-lock primitive keyed by cache key."""

    def try_acquire(self, key: str) -> bool:
        """Return True if the caller may compute. Never blocks."""
        ...

    def release(self, key: str) -> None:
        """Release a previously acquired lock."""
        ...


class ThreadingSingleFlight:
    """In-process single-flight using per-key ``threading.Lock`` instances.

    Entries are created on first acquisition and dropped on release, subject to
    a bounded map so a flood of distinct keys cannot grow memory without bound.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._max_entries = max_entries
        self._locks: dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        with self._meta:
            lock = self._locks.get(key)
            if lock is None:
                if len(self._locks) >= self._max_entries:
                    self._locks.clear()
                lock = threading.Lock()
                self._locks[key] = lock
            return lock.acquire(blocking=False)

    def release(self, key: str) -> None:
        with self._meta:
            lock = self._locks.get(key)
            if lock is None:
                return
            if lock.locked():
                lock.release()
            if not lock.locked():
                del self._locks[key]


class RedisSingleFlight:
    """Distributed single-flight via a ``SET NX PX`` lease.

    Contenders (whose ``try_acquire`` returns False) never hold a lock; they
    poll the cache value for the result while the leader computes. ``release``
    deletes the lock only if the lease token still matches, so a stale release
    from an expired lease cannot release a new owner's lock.
    """

    def __init__(
        self,
        client: object,
        lease_seconds: float = 5.0,
    ) -> None:
        self._client = _as_redis(client)
        self._lease_ms = max(1, int(lease_seconds * 1000))
        self._client.ping()
        self._release_script = self._client.register_script(_ACQUIRE_RELEASE_SCRIPT)
        self._salt = secrets.token_hex(8)

    def try_acquire(self, key: str) -> bool:
        token = self._token(key)
        acquired = self._client.set(self._lock_key(key), token, nx=True, px=self._lease_ms)
        return bool(acquired)

    def release(self, key: str) -> None:
        try:
            self._release_script(keys=[self._lock_key(key)], args=[self._token(key)])
        except Exception:  # noqa: BLE001 - releasing is best-effort
            pass

    def _lock_key(self, key: str) -> str:
        return f"{key}:lock"

    def _token(self, key: str) -> str:
        return f"{self._salt}:{threading.get_ident()}:{key}"


def _as_redis(client: object) -> object:
    return client


__all__ = ["SingleFlight", "ThreadingSingleFlight", "RedisSingleFlight"]