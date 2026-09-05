"""Prompt construction tests."""

from app.application.services.prompt import (
    build_system_prompt,
    build_user_prompt,
    citations_from_chunks,
)
from app.domain.entities import RetrievedChunk
from tests.fakes import make_chunk


def test_system_prompt_contains_grounding_rules() -> None:
    prompt = build_system_prompt()
    assert "provided context" in prompt.lower()
    assert "never fabricate" in prompt.lower()
    assert "untrusted" in prompt.lower()
    assert "ignore any instructions" in prompt.lower()
    assert "inference" in prompt.lower()


def test_user_prompt_structure() -> None:
    prompt = build_user_prompt("CONTEXT_BLOCK", "What is the policy?")
    assert "CONTEXT_BLOCK" in prompt
    assert "What is the policy?" in prompt
    assert "UNTRUSTED DATA" in prompt.upper()


def test_user_prompt_with_empty_context() -> None:
    prompt = build_user_prompt("", "question?")
    assert "No context was retrieved" in prompt


def test_citations_produced_in_order() -> None:
    chunks = []
    for cid in ["c1", "c2", "c3"]:
        chunk = make_chunk(chunk_id=cid, text=f"text {cid}", document_id=f"doc_{cid}")
        chunks.append(RetrievedChunk.from_chunk(chunk))
    citations = citations_from_chunks(chunks)
    assert [c.chunk_id for c in citations] == ["c1", "c2", "c3"]
    c = citations[0]
    assert c.document_id == "doc_c1"
    assert c.page_number == 1
    assert c.filename


def test_citations_dedupe() -> None:
    chunk = make_chunk(chunk_id="dup", text="same")
    chunks = [RetrievedChunk.from_chunk(chunk), RetrievedChunk.from_chunk(chunk)]
    citations = citations_from_chunks(chunks)
    assert len(citations) == 1