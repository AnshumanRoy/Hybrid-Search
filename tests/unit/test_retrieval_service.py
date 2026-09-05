"""RetrievalService tests (hybrid fake + fake reranker)."""

import pytest

from app.application.services.retrieval import RetrievalService
from app.domain.entities import RetrievedChunk
from app.domain.exceptions import RetrievalError
from app.domain.value_objects import SearchFilters
from tests.fakes import NoopReranker, make_chunk


class FakeHybrid:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k, *, semantic_top_k=None, lexical_top_k=None, filters=None, query_id=None):
        self.calls.append((query, top_k, query_id))
        return self.results[:top_k]


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk.from_chunk(make_chunk(chunk_id=cid, text=f"text {cid}"))


def test_retrieve_returns_deduplicated_top_k() -> None:
    service = RetrievalService(FakeHybrid([_chunk("a"), _chunk("b"), _chunk("a")]))
    out = service.retrieve("query", top_k=2)
    assert len(out) == 2


def test_retrieve_rejects_empty_query() -> None:
    service = RetrievalService(FakeHybrid([]))
    with pytest.raises(RetrievalError):
        service.retrieve("   ", top_k=5)


def test_retrieve_rejects_long_query() -> None:
    service = RetrievalService(FakeHybrid([]), max_query_length=10)
    with pytest.raises(RetrievalError):
        service.retrieve("x" * 11, top_k=5)


def test_reranker_applied_when_provided() -> None:
    service = RetrievalService(FakeHybrid([_chunk("a")]), reranker=NoopReranker())
    out = service.retrieve("q", top_k=3)
    assert len(out) == 1
    assert out[0].rerank_score is None  # noop leaves unchanged


def test_no_candidates_returns_empty() -> None:
    service = RetrievalService(FakeHybrid([]))
    assert service.retrieve("nothing to find", top_k=3) == []


def test_default_top_k_used() -> None:
    hybrid = FakeHybrid([_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")])
    service = RetrievalService(hybrid, default_top_k=2)
    out = service.retrieve("q")
    assert len(out) == 2