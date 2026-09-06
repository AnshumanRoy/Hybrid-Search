"""Unit tests for the fixed-window per-user ``RateLimiter``."""

from __future__ import annotations

import pytest

from app.application.services.rate_limiting import RateLimiter
from app.common.cache import InMemoryCache
from app.domain.exceptions import RateLimitExceededError


def make_limiter(*windows: tuple[int, int]) -> RateLimiter:
    return RateLimiter(InMemoryCache(), windows=windows)


def test_allows_under_limit() -> None:
    limiter = make_limiter((60, 3))
    for _ in range(3):
        limiter.consume("user-a", _now=1_000_000)  # same minute bucket (16666)


def test_blocks_at_limit_same_bucket() -> None:
    limiter = make_limiter((60, 2))
    limiter.consume("user-a", _now=1_000_000)
    limiter.consume("user-a", _now=1_000_001)
    with pytest.raises(RateLimitExceededError):
        limiter.consume("user-a", _now=1_000_002)


def test_resets_when_window_rolls_over() -> None:
    limiter = make_limiter((60, 1))
    limiter.consume("user-a", _now=1_000_000)  # bucket 16666 spans [999960, 1000020)
    with pytest.raises(RateLimitExceededError):
        limiter.consume("user-a", _now=1_000_019)  # still bucket 16666, at capacity
    limiter.consume("user-a", _now=1_000_020)  # next minute: fresh bucket 16667


def test_isolates_per_user() -> None:
    limiter = make_limiter((60, 1))
    limiter.consume("user-a", _now=1_000_000)
    with pytest.raises(RateLimitExceededError):
        limiter.consume("user-a", _now=1_000_001)
    limiter.consume("user-b", _now=1_000_001)  # unaffected


def test_daily_window_independent_of_burst_window() -> None:
    limiter = make_limiter((60, 1000), (86_400, 2))
    limiter.consume("user-a", _now=1_000_000)
    limiter.consume("user-a", _now=1_000_100)  # same daily bucket (11), wrong burst buckets
    with pytest.raises(RateLimitExceededError):
        limiter.consume("user-a", _now=1_000_200)


def test_no_cache_never_blocks() -> None:
    limiter = RateLimiter(None, windows=[(60, 1)])
    for _ in range(5):
        limiter.consume("user-a")


def test_no_windows_never_blocks() -> None:
    limiter = make_limiter()
    for _ in range(5):
        limiter.consume("user-a")