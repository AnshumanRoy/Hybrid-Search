"""Fusion algorithm tests (weighted + RRF)."""

from app.common.utils.fusion import rrf_fusion, weighted_fusion
from app.domain.entities import RetrievedChunk


def _chunk(cid: str, semantic: float | None = None, lexical: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=f"d_{cid}",
        filename="x.txt",
        text=f"text {cid}",
        semantic_score=semantic,
        lexical_score=lexical,
    )


def test_weighted_fusion_orientation() -> None:
    semantic = [_chunk("A", semantic=0.9, lexical=None), _chunk("B", semantic=0.1, lexical=None)]
    lexical = [_chunk("B", semantic=None, lexical=0.2), _chunk("A", semantic=None, lexical=0.1)]
    fused = weighted_fusion(semantic, lexical, weight_semantic=0.7)
    assert fused[0].chunk_id == "A"
    assert fused[0].fused_score is not None
    assert {c.chunk_id for c in fused} == {"A", "B"}


def test_weighted_fusion_dedupes() -> None:
    semantic = [_chunk("A", semantic=0.8)]
    lexical = [_chunk("A", lexical=0.3)]
    fused = weighted_fusion(semantic, lexical, weight_semantic=0.5)
    assert len(fused) == 1


def test_rrf_ranks_document_that_appears_higher_in_both() -> None:
    semantic = [_chunk("A", lexical=None), _chunk("B", lexical=None)]
    lexical = [_chunk("A", lexical=None), _chunk("B", lexical=None)]
    # A is rank 1 in both lists; should lead.
    fused = rrf_fusion(semantic, lexical, k=60)
    assert fused[0].chunk_id == "A"
    assert fused[0].fused_score > fused[1].fused_score


def test_rrf_handles_disjoint_sets() -> None:
    semantic = [_chunk("X")]
    lexical = [_chunk("Y")]
    fused = rrf_fusion(semantic, lexical, k=60)
    assert {c.chunk_id for c in fused} == {"X", "Y"}


def test_rrf_weights_affect_order() -> None:
    semantic = [_chunk("S", semantic=None), _chunk("L", semantic=None)]
    lexical = [_chunk("L", semantic=None), _chunk("S", semantic=None)]
    weighted_toward_lexical = rrf_fusion(semantic, lexical, k=10, semantic_weight=0.1, lexical_weight=10.0)
    assert weighted_toward_lexical[0].chunk_id == "L"


def test_fusion_returns_new_objects_preserving_metadata() -> None:
    sem = _chunk("A", semantic=0.9, lexical=None)
    fused = weighted_fusion([sem], [], weight_semantic=1.0)
    assert fused[0].chunk_id == "A"
    assert fused[0].retrieval_metadata.get("weighted_fusion") is not None