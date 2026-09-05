"""Unit tests for stdlib password hashing and signed bearer tokens."""

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
    payload_b64, sig = token.rsplit(".", 1)
    forged = base64.urlsafe_b64encode(json.dumps({"sub": "attacker"}).encode()).rstrip(b"=").decode()
    assert verify_token(f"{forged}.{sig}", "secret-key") is None


def test_token_rejects_tampered_signature() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=3600)
    payload_b64, _ = token.rsplit(".", 1)
    assert verify_token(f"{payload_b64}.AAAA", "secret-key") is None


def test_token_rejects_garbage() -> None:
    assert verify_token("not-a-token", "secret-key") is None
    assert verify_token("", "secret-key") is None


def test_token_expires() -> None:
    token = sign_token({"sub": "user-1"}, "secret-key", ttl_seconds=60, _now=1_000)
    assert verify_token(token, "secret-key", _now=1_030) is not None
    assert verify_token(token, "secret-key", _now=1_061) is None