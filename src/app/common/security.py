"""Password hashing and signed bearer tokens (stdlib only).

Passwords: PBKDF2-HMAC-SHA256 (OWASP-recommended parameters) with per-user
random salts, stored as ``pbkdf2_sha256$iterations$salt_b64$hash_b64``.

Tokens: URL-safe base64 ``payload.signature`` with an HMAC-SHA256 signature
over the base64 payload using a shared secret; expiry enforced on verify.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

_PBKDF2_ITERATIONS = 310_000
_HASH_ALGORITHM = "sha256"


def hash_password(password: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(_HASH_ALGORITHM, password.encode("utf-8"), salt, iterations)
    return "$".join(
        [
            f"pbkdf2_{_HASH_ALGORITHM}",
            str(iterations),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_s, salt_b64, hash_b64 = stored.split("$", 3)
        if not algorithm.startswith("pbkdf2_"):
            return False
        iterations = int(iterations_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    digest = hashlib.pbkdf2_hmac(_HASH_ALGORITHM, password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _signature(payload_b64: str, secret_key: str) -> str:
    return _b64url(hmac.new(secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest())


def sign_token(
    payload: dict[str, Any],
    secret_key: str,
    ttl_seconds: int,
    *,
    _now: float | None = None,
) -> str:
    body: dict[str, Any] = dict(payload)
    body["iat"] = int(_now if _now is not None else time.time())
    body["exp"] = body["iat"] + ttl_seconds
    payload_b64 = _b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_signature(payload_b64, secret_key)}"


def verify_token(token: str, secret_key: str, *, _now: float | None = None) -> dict[str, Any] | None:
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected = _signature(payload_b64, secret_key)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload_b64 + "==").decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
    now = _now if _now is not None else time.time()
    if payload.get("exp", 0) <= now:
        return None
    return payload