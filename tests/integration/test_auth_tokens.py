"""Token life-cycle integration tests: refresh, revocation, and rate limiting.

Rate limiting is exercised through a dedicated container with a tiny quota so
``/query`` faults with 429 without touching a real LLM backend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.bootstrap import build_container
from app.common.configuration.settings import Settings
from tests.fakes import FakeEmbedder, RecordingLLM

PASSWORD = "hunter2-password"


def _build_client(tmp_path, **settings_overrides) -> TestClient:
    settings = Settings(
        embedding_model="fake-token-model",
        vector_dimension=8,
        embedding_batch_size=4,
        vector_store_dir=tmp_path / "vs",
        database_url="sqlite://",
        data_dir=tmp_path / "data",
        chunk_size=120,
        chunk_overlap=20,
        auth_secret_key="test-secret",
        **settings_overrides,
    )
    container = build_container(
        settings,
        with_reranker=False,
        embedder=FakeEmbedder(model_name="fake-token-model"),
        vector_store=None,
        llm_providers={"ollama": RecordingLLM("ollama")},
    )
    return TestClient(create_app(container))


def register(client: TestClient, email: str = "alice@example.com") -> str:
    response = client.post(
        "/auth/register",
        json={"email": email, "password": PASSWORD, "display_name": "Alice"},
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_refresh_issues_new_token_and_old_still_valid(tmp_path) -> None:
    client = _build_client(tmp_path)
    token = register(client)
    refreshed = client.post("/auth/refresh", headers=auth(token))
    assert refreshed.status_code == 200, refreshed.text
    new_token = refreshed.json()["token"]
    assert new_token != token
    assert client.get("/auth/me", headers=auth(new_token)).status_code == 200


def test_revoke_invalidates_all_outstanding_tokens(tmp_path) -> None:
    client = _build_client(tmp_path)
    token1 = register(client)
    refreshed = client.post("/auth/refresh", headers=auth(token1))
    token2 = refreshed.json()["token"]

    revoked = client.post("/auth/revoke", headers=auth(token2))
    assert revoked.status_code == 204

    for old in (token1, token2):
        me = client.get("/auth/me", headers=auth(old))
        assert me.status_code == 401, me.text

    # Re-login re-issues a token at the new version.
    login = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert login.status_code == 200
    assert client.get("/auth/me", headers=auth(login.json()["token"])).status_code == 200


def test_rate_limit_exceeded_returns_429(tmp_path) -> None:
    client = _build_client(
        tmp_path,
        cache_enabled=True,
        rate_limit_window_seconds=60,
        rate_limit_max_requests=2,
        rate_limit_daily_max_requests=1000,
    )
    token = register(client)
    headers = auth(token)

    assert client.post("/query", headers=headers, json={"query": "hello", "top_k": 3}).status_code == 200
    assert client.post("/query", headers=headers, json={"query": "world", "top_k": 3}).status_code == 200
    blocked = client.post("/query", headers=headers, json={"query": "third", "top_k": 3})
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers

    # A different user is unaffected.
    other = register(client, "bob@example.com")
    assert client.post("/query", headers=auth(other), json={"query": "hi", "top_k": 3}).status_code == 200