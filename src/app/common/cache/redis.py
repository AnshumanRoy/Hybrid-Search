"""Redis-backed ``Cache`` implementation.

Values are JSON-encoded. Pydantic models and datetimes are handled via the
default fallback, so both the plain-string payloads used by ``LLMService`` and
the envelope strings produced by ``CachingVectorStore`` round-trip unchanged.
TTL is applied via Redis ``EX``. All keys are namespaced by a constructor
``prefix`` so ``clear()`` never touches unrelated keys.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RedisCache:
    """Generic TTL cache backed by Redis (the ``Cache`` protocol)."""

    def __init__(self, client: object, prefix: str = "vc:", ttl_seconds: int | None = None) -> None:
        self._client = _as_redis(client)
        self._prefix = prefix
        self._default_ttl = ttl_seconds
        self._client.ping()

    @property
    def backend(self) -> str:
        return "redis"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(f"{self._prefix}{key}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("redis cache value for key=%s is not valid JSON", key)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        try:
            payload = json.dumps(value, default=_json_default)
        except (TypeError, ValueError) as exc:
            logger.warning("redis cache failed to serialize key=%s: %s", key, exc)
            return
        if ttl is not None and ttl > 0:
            self._client.set(f"{self._prefix}{key}", payload, ex=ttl)
        else:
            self._client.set(f"{self._prefix}{key}", payload)

    def delete(self, key: str) -> None:
        self._client.delete(f"{self._prefix}{key}")

    def clear(self) -> None:
        keys = list(self._scan(f"{self._prefix}*"))
        if keys:
            self._client.delete(*keys)

    def _scan(self, pattern: str) -> list[str]:
        found: list[str] = []
        cursor = 0
        while True:
            cursor, batch = self._client.scan(cursor, match=pattern, count=200)
            found.extend(batch)
            if cursor == 0:
                return [k.decode("utf-8") if isinstance(k, bytes) else k for k in found]


def _json_default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError(f"cannot serialize {type(o).__name__}")


def _as_redis(client: object) -> object:
    return client


__all__ = ["RedisCache"]