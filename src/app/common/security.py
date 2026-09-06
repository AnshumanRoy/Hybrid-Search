"""Password hashing and JWT (HS256) signing/verification — stdlib only.

Passwords: PBKDF2-HMAC-SHA256 (OWASP-recommended parameters) with per-user
random salts, stored as ``pbkdf2_sha256$iterations$salt_b64$hash_b64``.

Tokens: standard JWT compact serialization ``header.payload.signature`` with
the JOSE header ``{"alg":"HS256","typ":"JWT"}``. The signature is an
HMAC-SHA256 of the signing input (``header.payload``) under a shared secret;
``exp`` is enforced on verify. ALGORITHM IS PINNED at ``HS256`` during
verification regardless of what the header claims, so ``alg=none``-style
confusion attacks are rejected out of the box.
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

_JWT_ALGORITHM = "HS256"
_JWT_TYPE = "JWT"


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


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(f"{segment}{padding}")


def _jose_header_b64() -> str:
    return _b64url(json.dumps({"alg": _JWT_ALGORITHM, "typ": _JWT_TYPE}, separators=(",", ":")).encode("utf-8"))


def _signature(signing_input: str, secret_key: str) -> str:
    return _b64url(hmac.new(secret_key.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest())


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
    header_b64 = _jose_header_b64()
    payload_b64 = _b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}"
    return f"{signing_input}.{_signature(signing_input, secret_key)}"


def verify_token(token: str, secret_key: str, *, _now: float | None = None) -> dict[str, Any] | None:
    segments = token.split(".")
    if len(segments) != 3:
        return None
    header_b64, payload_b64, sig = segments
    signing_input = f"{header_b64}.{payload_b64}"
    if not hmac.compare_digest(sig, _signature(signing_input, secret_key)):
        return None

    # Decode the header strictly and PIN the algorithm. A valid signature over a
    # forged "alg":"none" header must still be rejected.
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        if header.get("alg") != _JWT_ALGORITHM:
            return None
    except (ValueError, TypeError, UnicodeDecodeError):
        return None

    try:
        payload: dict[str, Any] = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError):
        return None

    now = _now if _now is not None else time.time()
    if payload.get("exp", 0) <= now:
        return None
    return payload