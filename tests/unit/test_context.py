"""Context construction tests."""

from app.application.services.context import ContextBuilder
from app.domain.entities import RetrievedChunk
from tests.fakes import make_chunk


def _chunk(cid: str, text: str, page: int = 1, filename: str = "policy.pdf") -> RetrievedChunk:
    chunk = make_chunk(chunk_id=cid, text=text, page=page, filename="policy.pdf", document_id="doc_" + filename)
    return RetrievedChunk.from_chunk(chunk)


def test_build_formats_sources() -> None:
    chunks = [_chunk("a", "Alpha text."), _chunk("b", "Beta text.", page=3)]
    builder = ContextBuilder()
    out = builder.build(chunks)
    assert "[Source 1]\nDocument: policy.pdf\nPage: 1" in out
    assert "[Source 2]\nDocument: policy.pdf\nPage: 3" in out
    assert "Alpha text." in out
    assert "Beta text." in out


def test_duplicates_removed() -> None:
    chunks = [_chunk("a", "same chunk"), _chunk("a", "same chunk")]
    out = ContextBuilder().build(chunks)
    assert out.count("[Source") == 1


def test_budget_is_enforced() -> None:
    chunks = [_chunk("a", "A" * 900), _chunk("b", "B" * 900)]
    out = ContextBuilder(max_chars=1000).build(chunks)
    assert "B" not in out


def test_empty_drop() -> None:
    empty = RetrievedChunk.from_chunk(make_chunk(chunk_id="e", text=""))
    out = ContextBuilder().build([empty])
    assert out == ""


def test_empty_chunks_produce_marker() -> None:
    assert ContextBuilder().build([]) == ""


def test_invalid_budget_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        ContextBuilder(max_chars=10)