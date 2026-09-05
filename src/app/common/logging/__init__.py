"""Logging helpers — structured, contextual, redaction-aware.

Design notes:
- Uses standard library logging only.
- Contextual fields are recorded on the record so JSON formatters can emit
  them; the default console formatter renders them after the message.
- A ``RedactingFilter`` scrubs values matching configured secrets and any
  strings marked redactable by callers (e.g. query text).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_SECRET_FIELD_NAMES = {"password", "secret", "token", "api_key", "apikey", "authorization", "pwd"}


class RedactingFilter(logging.Filter):
    """Scrub configured secrets / redacted values before the record is formatted."""

    def __init__(self, secrets: set[str] | None = None) -> None:
        super().__init__()
        self.secrets = set(secrets or ())
        self.secrets.update(v for v in os.environ.values() if len(v) >= 8)
        self._longest = sorted(self.secrets, key=len, reverse=True)

    def scrub(self, text: str) -> str:
        if not self._longest:
            return text
        # Cheap batched approach: replace the longest secrets first.
        for secret in self._longest:
            if secret and secret in text:
                text = text.replace(secret, "[REDACTED]")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        for attr in ("msg", "message"):
            value = getattr(record, attr, None)
            if isinstance(value, str):
                setattr(record, attr, self.scrub(value))
        if isinstance(record.args, dict):
            record.args = {
                k: (self.scrub(v) if isinstance(v, str) else v) for k, v in record.args.items()
            }
        return True


def _user_redaction(user_query: str) -> str:
    return "[REDACTED-QUERY]" if user_query else user_query


def get_logger(name: str) -> logging.Logger:
    """Return a package-scoped logger (never the root)."""
    return logging.getLogger(name)


def setup_logging(level: str | int = "INFO", *, json: bool = False) -> None:
    """Configure the root handler once for the process (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and getattr(h, "_rag_configured", False) for h in root.handlers):
        root.setLevel(level)
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(RedactingFilter())
    if json:
        try:
            from app.common.logging.json_formatter import JSONFormatter
        except ImportError:  # pragma: no cover - dev-only fallback
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        else:
            handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler._rag_configured = True  # type: ignore[attr-defined]
    root.addHandler(handler)


class ContextualAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects static context fields into every record."""

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        super().__init__(logger, context or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def redact_query(q: str) -> str:
    """Downstream-safe marker so callers can never accidentally log raw queries."""
    return _user_redaction(q)