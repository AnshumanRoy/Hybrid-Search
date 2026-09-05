"""Document CRUD + sharing access-control flows (multi-user)."""

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
        embedding_model="fake-access-model",
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
        embedder=FakeEmbedder(model_name="fake-access-model"),
        vector_store=None,
    )
    return TestClient(create_app(container))


def register(client: TestClient, email: str) -> str:
    response = client.post(
        "/auth/register",
        json={"email": email, "password": PASSWORD, "display_name": email.split("@")[0]},
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def ingest(client: TestClient, token: str, filename: str, text: str):
    return client.post(
        "/documents/ingest",
        headers=auth(token),
        files={"file": (filename, text.encode(), "text/plain")},
    )


def test_edit_permission_and_share_flow(client) -> None:
    alice = register(client, "alice@example.com")
    bob = register(client, "bob@example.com")

    ingest(
        client,
        alice,
        "report.txt",
        "Alpha report. Quarterly revenue grew three percent quarter over quarter.",
    )
    doc_list = client.get("/documents/", headers=auth(alice)).json()
    doc_id = doc_list[0]["document_id"]
    assert doc_list[0]["permission"] == "owner"

    # Bob cannot see or fetch Alice's private document (404, no existence leak).
    assert client.get("/documents/", headers=auth(bob)).json() == []
    assert client.get(f"/documents/{doc_id}", headers=auth(bob)).status_code == 404

    # Bob cannot delete, update or share a document he cannot see.
    assert client.delete(f"/documents/{doc_id}", headers=auth(bob)).status_code == 404
    assert client.put(f"/documents/{doc_id}", headers=auth(bob),
                      files={"file": ("x.txt", b"hax", "text/plain")}).status_code == 404
    assert client.post(
        f"/documents/{doc_id}/share",
        headers=auth(bob),
        json={"user_email": "carol@example.com", "permission": "viewer"},
    ).status_code == 404

    # Alice shares as viewer → Bob can read but not edit.
    share = client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "bob@example.com", "permission": "viewer"},
    )
    assert share.status_code == 200
    assert share.json()["permission"] == "viewer"

    bob_doc = client.get(f"/documents/{doc_id}", headers=auth(bob))
    assert bob_doc.status_code == 200
    assert bob_doc.json()["permission"] == "viewer"
    assert bob_doc.json()["shared"] is True
    assert client.put(
        f"/documents/{doc_id}", headers=auth(bob), files={"file": ("x.txt", b"hax", "text/plain")}
    ).status_code == 403

    # Escalate to editor → Bob can re-index.
    client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "bob@example.com", "permission": "editor"},
    )
    update = client.put(
        f"/documents/{doc_id}",
        headers=auth(bob),
        files={"file": ("report.txt", b"Alpha report. Revised content for the editor.", "text/plain")},
    )
    assert update.status_code == 200, update.text
    assert update.json()["ingest_status"] == "indexed"

    # Editor cannot delete; owner can.
    assert client.delete(f"/documents/{doc_id}", headers=auth(bob)).status_code == 403
    assert client.delete(f"/documents/{doc_id}", headers=auth(alice)).status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=auth(alice)).status_code == 404


def test_shares_endpoints_owner_only(client) -> None:
    alice = register(client, "alice@example.com")
    bob = register(client, "bob@example.com")
    carol = register(client, "carol@example.com")

    ingest(client, alice, "doc.txt", "sharing endpoints test content.")
    doc_id = client.get("/documents/", headers=auth(alice)).json()[0]["document_id"]

    client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "bob@example.com", "permission": "viewer"},
    )
    client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "carol@example.com", "permission": "editor"},
    )

    shares = client.get(f"/documents/{doc_id}/shares", headers=auth(alice))
    assert shares.status_code == 200
    assert {s["user_email"]: s["permission"] for s in shares.json()} == {
        "bob@example.com": "viewer",
        "carol@example.com": "editor",
    }

    # Non-owner listing shares is forbidden.
    assert client.get(f"/documents/{doc_id}/shares", headers=auth(bob)).status_code == 403

    # Unshare revokes access.
    assert client.delete(f"/documents/{doc_id}/share/bob@example.com", headers=auth(alice)).status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=auth(bob)).status_code == 404


def test_sharing_with_unregistered_user_fails(client) -> None:
    alice = register(client, "alice@example.com")
    ingest(client, alice, "doc.txt", "share target must exist.")
    doc_id = client.get("/documents/", headers=auth(alice)).json()[0]["document_id"]

    response = client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "ghost@example.com", "permission": "viewer"},
    )
    assert response.status_code == 404


def test_public_documents_readable_by_everyone(client) -> None:
    from datetime import datetime, timezone

    from app.infrastructure.persistence.models import ChunkRow, DocumentRow

    alice = register(client, "alice@example.com")
    bob = register(client, "bob@example.com")

    # Seed an ownerless (legacy/public) document directly in the DB, simulating
    # rows created before the owner_id migration.
    container = client.app.state.container
    with container.database.session_factory() as session:
        now = datetime.now(timezone.utc)
        doc = DocumentRow(
            document_id="public_doc_id",
            filename="public_note.txt",
            content_hash="public-hash",
            metadata_json={},
            page_count=1,
            status="indexed",
            created_at=now,
            updated_at=now,
            owner_id=None,
        )
        session.add(doc)
        session.add(
            ChunkRow(
                chunk_id="public_chunk_id",
                document_id="public_doc_id",
                filename="public_note.txt",
                page_number=1,
                text="public note everyone can read",
                chunk_index=0,
                metadata_json={},
            )
        )
        session.commit()

    public_id = client.get("/documents/", headers=auth(bob)).json()[0]["document_id"]
    gotten = client.get(f"/documents/{public_id}", headers=auth(bob))
    assert gotten.status_code == 200
    assert gotten.json()["permission"] == "viewer"
    assert gotten.json()["owner_id"] is None

    # Public docs have no owner → nobody can share or delete them.
    assert client.post(
        f"/documents/{public_id}/share",
        headers=auth(bob),
        json={"user_email": "alice@example.com", "permission": "viewer"},
    ).status_code == 403
    assert client.delete(f"/documents/{public_id}", headers=auth(bob)).status_code == 403


def test_update_reindexes_in_place_and_keeps_shares(client) -> None:
    alice = register(client, "alice@example.com")
    bob = register(client, "bob@example.com")

    ingest(client, alice, "report.txt", "version one content.")
    doc_id = client.get("/documents/", headers=auth(alice)).json()[0]["document_id"]
    client.post(
        f"/documents/{doc_id}/share",
        headers=auth(alice),
        json={"user_email": "bob@example.com", "permission": "viewer"},
    )

    update = client.put(
        f"/documents/{doc_id}",
        headers=auth(alice),
        files={"file": ("report.txt", b"version two with different bytes here.", "text/plain")},
    )
    assert update.status_code == 200, update.text
    assert update.json()["document_id"] == doc_id  # deterministic id → in-place re-index
    assert update.json()["ingest_status"] == "indexed"

    # Shares survive re-index and the new content is searchable for Bob.
    assert client.get(f"/documents/{doc_id}", headers=auth(bob)).status_code == 200
    query = client.post(
        "/query",
        headers=auth(bob),
        json={"query": "version two", "top_k": 5},
    )
    assert query.status_code == 200
    texts = [c["text"] for c in query.json()["retrieved_chunks"]]
    assert any("version two" in t for t in texts)
    assert not any("version one" in t for t in texts)