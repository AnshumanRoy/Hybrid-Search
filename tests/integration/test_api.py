"""FastAPI endpoint tests — run against an app wired with fakes (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.bootstrap import build_container
from app.common.configuration.settings import Settings
from tests.fakes import FakeEmbedder

PASSWORD = "hunter2-password"


@pytest.fixture
def client(tmp_path):
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
    )
    container = build_container(
        settings,
        with_reranker=False,
        embedder=FakeEmbedder(model_name="fake-api-model"),
        vector_store=None,  # real Chroma against tmp dir
    )
    app = create_app(container)
    return TestClient(app)


def register(client: TestClient, email: str = "alice@example.com", password: str = PASSWORD) -> str:
    """Register a user and return a bearer token."""
    response = client.post(
        "/auth/register",
        json={"email": email, "password": password, "display_name": "Alice"},
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_unauthenticated(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_requires_auth(client) -> None:
    response = client.post("/query", json={"query": "hello", "top_k": 5})
    assert response.status_code == 401


def test_ingest_upload_then_query(client) -> None:
    token = register(client)
    headers = auth(token)
    data = (
        b"Acme Corporation refund policy. Customers may return items within "
        b"30 days for a full refund. Digital downloads are non-refundable."
    )
    ingest = client.post(
        "/documents/ingest",
        headers=headers,
        files={"file": ("policy.txt", data, "text/plain")},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    assert payload[0]["status"] == "indexed"
    assert payload[0]["chunks_embedded"] >= 1

    query = client.post(
        "/query",
        headers=headers,
        json={"query": "What is the refund policy?", "top_k": 5},
    )
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert body["citations"][0]["document"] == "policy.txt"


def test_query_rejects_oversized_top_k(client) -> None:
    token = register(client)
    response = client.post("/query", headers=auth(token), json={"query": "hello", "top_k": 1000})
    assert response.status_code == 422


def test_query_rejects_empty_query(client) -> None:
    token = register(client)
    response = client.post("/query", headers=auth(token), json={"query": ""})
    assert response.status_code == 422


def test_ingest_rejects_empty_upload(client) -> None:
    token = register(client)
    response = client.post(
        "/documents/ingest",
        headers=auth(token),
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_list_documents_shows_only_your_scope(client) -> None:
    alice = auth(register(client, "alice@example.com"))
    bob = auth(register(client, "bob@example.com"))
    data = b"single sentence to ingest for listing."
    client.post("/documents/ingest", headers=alice, files={"file": ("a.txt", data, "text/plain")})

    alice_list = client.get("/documents/", headers=alice)
    assert alice_list.status_code == 200
    assert [d["filename"] for d in alice_list.json()] == ["a.txt"]

    # Bob does not share Alice's private document.
    bob_list = client.get("/documents/", headers=bob)
    assert [d["filename"] for d in bob_list.json()] == []


def test_login_and_me(client) -> None:
    token = register(client)
    login = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert login.status_code == 200
    assert login.json()["user"]["email"] == "alice@example.com"

    me = client.get("/auth/me", headers=auth(token))
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


def test_login_rejects_bad_password(client) -> None:
    register(client)
    response = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrong"})
    assert response.status_code == 401


def test_register_duplicate_email(client) -> None:
    register(client)
    response = client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": PASSWORD},
    )
    assert response.status_code == 409