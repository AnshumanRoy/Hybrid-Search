"""Endpoint tests for /rewrite and /query rewriting — fakes only, no network.

Focused on the contract: auth required, no-retrieval side effects, strategy
selection, clamping, and the metadata contract on /query responses.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.bootstrap import build_container
from app.common.configuration.settings import RewriteStrategy, Settings
from tests.fakes import FakeEmbedder, RecordingLLM

PASSWORD = "hunter2-password"


class StubRewriteLLM:
    """Registry fake that returns deterministic rewrite lines."""

    @property
    def model_name(self) -> str:
        return "nemotron"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        return f"variant line A for {prompt!r}\nvariant line B for {prompt!r}\n"


def _build_client(tmp_path, *, enabled: bool, strategy: RewriteStrategy) -> TestClient:
    settings = Settings(
        embedding_model="fake-api-model",
        vector_dimension=8,
        embedding_batch_size=4,
        vector_store_dir=tmp_path / "vs",
        database_url="sqlite://",
        data_dir=tmp_path / "data",
        chunk_size=120,
        chunk_overlap=20,
        cache_enabled=False,
        auth_secret_key="test-secret",
        nemotron_enabled=False,
        query_rewrite_enabled=enabled,
        query_rewrite_strategy=strategy,
    )
    container = build_container(
        settings,
        with_reranker=False,
        embedder=FakeEmbedder(model_name="fake-api-model"),
        vector_store=None,
        llm_providers={
            "ollama": RecordingLLM("ollama"),
            "nemotron": StubRewriteLLM(),
        },
    )
    return TestClient(create_app(container))


@pytest.fixture
def heuristic_client(tmp_path):
    return _build_client(tmp_path, enabled=True, strategy=RewriteStrategy.HEURISTIC)


@pytest.fixture
def llm_client(tmp_path):
    return _build_client(tmp_path, enabled=True, strategy=RewriteStrategy.LLM)


@pytest.fixture
def disabled_client(tmp_path):
    return _build_client(tmp_path, enabled=False, strategy=RewriteStrategy.LLM)


def register(client: TestClient) -> str:
    response = client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": PASSWORD, "display_name": "Alice"},
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestRewriteEndpoint:
    def test_requires_auth(self, heuristic_client):
        response = heuristic_client.post("/rewrite", json={"query": "q", "max_variants": 3})
        assert response.status_code == 401

    def test_heuristic_strategy_no_retrieval_side_effects(self, heuristic_client):
        token = register(heuristic_client)
        response = heuristic_client.post(
            "/rewrite",
            json={"query": "what is the refund policy", "strategy": "heuristic", "max_variants": 3},
            headers=auth(token),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["original"] == "what is the refund policy"
        assert body["variants"][0] == body["original"]
        assert body["strategy"] == "heuristic"
        assert len(body["variants"]) >= 1
        # Works against an entirely empty index -> no retrieval happened.

    def test_llm_strategy_uses_registered_backend_no_network(self, llm_client):
        token = register(llm_client)
        response = llm_client.post(
            "/rewrite",
            json={"query": "original query", "strategy": "llm", "max_variants": 3},
            headers=auth(token),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "llm"
        assert body["variants"][0] == "original query"
        assert len(body["variants"]) > 1

    def test_max_variants_clamped(self, llm_client):
        token = register(llm_client)
        response = llm_client.post(
            "/rewrite",
            json={"query": "q", "max_variants": 999},
            headers=auth(token),
        )
        assert response.status_code == 200
        assert len(response.json()["variants"]) <= 10

    def test_strategy_defaults_from_settings(self, heuristic_client):
        token = register(heuristic_client)
        response = heuristic_client.post("/rewrite", json={"query": "the refund policy"}, headers=auth(token))
        assert response.status_code == 200
        assert response.json()["strategy"] == "heuristic"


class TestQueryRewriting:
    def test_disabled_rewrite_true_silently_ignored(self, disabled_client):
        token = register(disabled_client)
        response = disabled_client.post(
            "/query",
            json={"query": "hello", "top_k": 3, "rewrite": True},
            headers=auth(token),
        )
        assert response.status_code == 200, response.text
        metadata = response.json()["metadata"]
        assert "rewritten_variants" not in metadata
        assert "rewrite_strategy" not in metadata

    def test_enabled_adds_rewrite_metadata(self, heuristic_client):
        token = register(heuristic_client)
        response = heuristic_client.post(
            "/query",
            json={"query": "what is the refund policy", "top_k": 3},
            headers=auth(token),
        )
        assert response.status_code == 200, response.text
        metadata = response.json()["metadata"]
        assert metadata["rewritten_variants"]
        assert metadata["rewrite_strategy"] == "heuristic"

    def test_enabled_false_override_drops_rewrite(self, heuristic_client):
        token = register(heuristic_client)
        response = heuristic_client.post(
            "/query",
            json={"query": "hello", "top_k": 3, "rewrite": False},
            headers=auth(token),
        )
        assert response.status_code == 200, response.text
        metadata = response.json()["metadata"]
        assert "rewritten_variants" not in metadata
        assert "rewrite_strategy" not in metadata