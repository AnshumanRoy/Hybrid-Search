"""Unit tests for stdlib password hashing and JWT (HS256) tokens."""

from __future__ import annotations

import base64
import json

from app.common.security import hash_password, sign_token, verify_password, verify_token


def test_password_hash_roundtrip() -> None:
    stored = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong password", stored)
    assert not verify_password("", stored)


def test_password_hash_format_and_iterations() -> None:
    stored = hash_password("x")
    assert stored.startswith("pbkdf2_sha256$310000$")


def test_password_hashes_differ_per_salt() -> None:
    assert hash_password("same password") != hash_password("same password")


def _parts(token: str) -> tuple[str, str, str]:
    header_b64, payload_b64, sig = token.split(".")
    return header_b64, payload_b64, sig


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def test_token_is_standard_jwt() -> None:
    token = sign_token({"sub": "user-1", "email": "a@example.com"}, "secret-key", ttl_seconds=3600)
    assert token.count(".") == 2
    header_b64, payload_b64, _sig = _parts(token)
    header = json.loads(base64.urlsafe_b64decode(header_b64 + "=="))
    assert header == {"alg": "HS256", "typ": "JWT"}
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
    assert payload["sub"] == "user-1"
    assert payload["email"] == "a@example.com"
    assert payload["exp"] > payload["iat"]


def test_token_roundtrip() -> None:
    payload = {"sub": "user-1", "email": "a@example.com"}
    token = sign_token(payload, "secret-key", ttl_seconds=3600)
    verified = verify_token(token, "secret-key")
    assert verified is not None
    assert verified["sub"] == "user-1"
    assert verified["email"] == "a@example.com"
    assert verified["exp"] > verified["iat"]


def test_token_rejects_wrong_secret() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=3600)
    assert verify_token(token, "other-key") is None


def test_token_rejects_tampered_payload() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=3600)
    header_b64, _payload_b64, sig = _parts(token)
    forged = _b64url(json.dumps({"sub": "attacker"}).encode())
    assert verify_token(f"{header_b64}.{forged}.{sig}", "secret-key") is None


def test_token_rejects_tampered_header() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=3600)
    _header_b64, payload_b64, sig = _parts(token)
    forged_header = _b64url(json.dumps({"alg": "HS384", "typ": "JWT"}).encode())
    assert verify_token(f"{forged_header}.{payload_b64}.{sig}", "secret-key") is None


def test_token_rejects_tampered_signature() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=3600)
    header_b64, payload_b64, _sig = _parts(token)
    assert verify_token(f"{header_b64}.{payload_b64}.AAAA", "secret-key") is None


def test_token_rejects_alg_none_confusion() -> None:
    # Even with a "valid" signature over an alg=none header, the pinned HS256
    # verification must reject the token.
    header_b64 = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload_b64 = _b64url(json.dumps({"sub": "user-1"}).encode())
    signing_input = f"{header_b64}.{payload_b64}"
    sig = _b64url(
        __import__("hashlib").new("sha256", signing_input.encode()).digest()  # not an HMAC signature
    )
    assert verify_token(f"{header_b64}.{payload_b64}.{sig}", "secret-key") is None


def test_token_rejects_legacy_two_segment_format() -> None:
    # The pre-JWT custom format was `payload.signature`; it must no longer verify.
    assert verify_token("cGF5bG9hZA.some-signature", "secret-key") is None


def test_token_rejects_garbage() -> None:
    assert verify_token("not-a-token", "secret-key") is None
    assert verify_token("", "secret-key") is None


def test_token_expires() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=60, _now=1_000)
    assert verify_token(token, "secret-key", _now=1_030) is not None
    assert verify_token(token, "secret-key", _now=1_061) is None