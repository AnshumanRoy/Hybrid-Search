"""Reranker behavior tests (protocol conformance via a scripted fake + sigmoid)."""

import pytest

from app.domain.entities import RetrievedChunk
from app.domain.interfaces import Reranker
from app.infrastructure.rerankers.cross_encoder import _sigmoid
from tests.fakes import NoopReranker, make_chunk


def _candidate(cid: str, text: str) -> RetrievedChunk:
    chunk = make_chunk(chunk_id=cid, text=text)
    return RetrievedChunk.from_chunk(chunk, fused_score=1.0)


def test_noop_reranker_conforms_and_preserves_metadata() -> None:
    reranker: Reranker = NoopReranker()
    candidates = [_candidate("a", "alpha"), _candidate("b", "beta")]
    out = reranker.rerank("query", candidates)
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert out[0].retrieval_metadata is not None


def test_noop_reranker_honors_top_k() -> None:
    reranker = NoopReranker()
    candidates = [_candidate("a", "alpha"), _candidate("b", "beta"), _candidate("c", "gamma")]
    out = reranker.rerank("query", candidates, top_k=2)
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_sigmoid_maps_to_unit_interval() -> None:
    assert 0.0 < _sigmoid(-5) < _sigmoid(0) < _sigmoid(5) < 1.0
    assert _sigmoid(0) == pytest.approx(0.5)


def test_reranker_rejects_empty() -> None:
    assert NoopReranker().rerank("q", []) == []