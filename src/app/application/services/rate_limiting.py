"""Per-user fixed-window rate limiting for cost-bearing endpoints.

The counter lives in the shared ``Cache`` backend (memory or Redis), keyed by
``user_id`` per window bucket so keys self-expire when a window rolls over —
no janitor. Multiple windows can be enforced at once (e.g. a 60s burst cap and
a daily quota) and every configured window must pass for a request to proceed.

Failure semantics match the rest of the cache layer: a missing/erroring cache
fails **open** (requests proceed) because rate limiting must never take the
API down; it is a cost guardrail, not a correctness mechanism.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from app.common.cache import Cache
from app.domain.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-window per-user limiter backed by the ``Cache`` protocol."""

    def __init__(
        self,
        cache: Cache | None,
        *,
        windows: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """``windows`` is a list of ``(window_seconds, max_requests)`` pairs."""
        self._cache = cache
        self._windows = list(windows or [])

    def consume(self, user_id: str, *, _now: float | None = None) -> None:
        """Record one request for ``user_id``; raise if any window is exceeded.

        Raises ``RateLimitExceededError`` when a window is already at capacity —
        the request is *not* counted on top of a full window (no compounding).
        """
        if self._cache is None or not self._windows:
            return
        now = _now if _now is not None else time.time()
        for window_seconds, max_requests in self._windows:
            bucket = int(now) // window_seconds
            key = f"rl:{user_id}:{window_seconds}:{bucket}"
            count = self._get_count(key)
            if count >= max_requests:
                raise RateLimitExceededError(
                    f"rate limit exceeded (max {max_requests} requests per {window_seconds}s window); "
                    "try again later"
                )
            self._bump_count(key, window_seconds)

    def _get_count(self, key: str) -> int:
        try:
            value = self._cache.get(key)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - limiter must never break requests
            logger.warning("rate limiter read failed; allowing request: %s", exc)
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _bump_count(self, key: str, window_seconds: int) -> None:
        try:
            current = int(self._cache.get(key) or 0)  # type: ignore[union-attr]
            self._cache.set(key, current + 1, ttl_seconds=window_seconds)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("rate limiter write failed; continuing: %s", exc)


__all__ = ["RateLimiter"]