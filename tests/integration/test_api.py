"""FastAPI endpoint tests — run against an app wired with fakes (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.bootstrap import build_container
from app.common.configuration.settings import Settings
from tests.fakes import FakeEmbedder


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
    )
    container = build_container(
        settings,
        with_reranker=False,
        embedder=FakeEmbedder(model_name="fake-api-model"),
        vector_store=None,  # real Chroma against tmp dir
    )
    app = create_app(container)
    return TestClient(app)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_ingest_upload_then_query(client) -> None:
    data = (
        b"Acme Corporation refund policy. Customers may return items within "
        b"30 days for a full refund. Digital downloads are non-refundable."
    )
    ingest = client.post(
        "/documents/ingest",
        files={"file": ("policy.txt", data, "text/plain")},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    assert payload[0]["status"] == "indexed"
    assert payload[0]["chunks_embedded"] >= 1

    query = client.post("/query", json={"query": "What is the refund policy?", "top_k": 5})
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert body["citations"][0]["document"] == "policy.txt"


def test_query_rejects_oversized_top_k(client) -> None:
    response = client.post("/query", json={"query": "hello", "top_k": 1000})
    assert response.status_code == 422


def test_query_rejects_empty_query(client) -> None:
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_ingest_rejects_empty_upload(client) -> None:
    response = client.post("/documents/ingest", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_list_documents(client) -> None:
    data = b"single sentence to ingest for listing."
    client.post("/documents/ingest", files={"file": ("a.txt", data, "text/plain")})
    response = client.get("/documents/")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["filename"] == "a.txt"