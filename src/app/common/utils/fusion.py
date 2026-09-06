"""Fusion algorithms for combining semantic + lexical result sets.

Pure functions — no I/O, no framework dependencies, easily unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.domain.entities import RetrievedChunk


def weighted_fusion(
    semantic: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    weight_semantic: float,
    weight_lexical: float | None = None,
) -> list[RetrievedChunk]:
    """Combine two ranked lists by a weighted sum of per-lists scores.

    Scores within each list are normalized to [0, 1].  Missing lists
    contribute zero.  Result is sorted descending by fused score (stable).
    """
    if weight_lexical is None:
        weight_lexical = 1.0 - weight_semantic

    semantic_norm = _normalize_scores([c.semantic_score for c in semantic])
    lexical_norm = _normalize_scores([c.lexical_score for c in lexical])

    semantic_by_id = {c.chunk_id: semantic_norm[i] for i, c in enumerate(semantic)}
    lexical_by_id = {c.chunk_id: lexical_norm[i] for i, c in enumerate(lexical)}
    semantic_base = {c.chunk_id: c for c in semantic}
    lexical_base = {c.chunk_id: c for c in lexical}

    ids = {c.chunk_id for c in semantic} | {c.chunk_id for c in lexical}

    fused: list[RetrievedChunk] = []
    for chunk_id in ids:
        base = semantic_base.get(chunk_id) or lexical_base.get(chunk_id)
        if base is None:
            continue
        score = weight_semantic * semantic_by_id.get(chunk_id, 0.0) + weight_lexical * lexical_by_id.get(chunk_id, 0.0)
        fused.append(
            base.model_copy(
                deep=True,
                update={
                    "fused_score": round(score, 6),
                    "retrieval_metadata": {
                        **base.retrieval_metadata,
                        "weighted_fusion": {
                            "weight_semantic": weight_semantic,
                            "weight_lexical": weight_lexical,
                            "semantic_norm": semantic_by_id.get(chunk_id),
                            "lexical_norm": lexical_by_id.get(chunk_id),
                        },
                    },
                },
            )
        )
    return sorted(fused, key=lambda c: (c.fused_score or 0.0), reverse=True)


def rrf_fusion(
    semantic: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    k: int = 60,
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion with per-list weights."""
    semantic_ids = [c.chunk_id for c in semantic]
    lexical_ids = [c.chunk_id for c in lexical]
    semantic_base = {c.chunk_id: c for c in semantic}
    lexical_base = {c.chunk_id: c for c in lexical}

    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(semantic_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + semantic_weight / (k + rank + 1)
    for rank, chunk_id in enumerate(lexical_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + lexical_weight / (k + rank + 1)

    all_ids = list(semantic_ids + [c for c in lexical_ids if c not in semantic_ids])
    all_ids.sort(key=lambda cid: scores.get(cid, 0.0), reverse=True)

    merged: list[RetrievedChunk] = []
    for chunk_id in all_ids:
        base = semantic_base.get(chunk_id) or lexical_base.get(chunk_id)
        if base is None:
            continue
        merged.append(
            base.model_copy(
                deep=True,
                update={
                    "fused_score": round(scores[chunk_id], 6),
                    "retrieval_metadata": {
                        **base.retrieval_metadata,
                        "rrf": {
                            "k": k,
                            "semantic_weight": semantic_weight,
                            "lexical_weight": lexical_weight,
                            "rrf_score": scores[chunk_id],
                        },
                    },
                },
            )
        )
    return merged


def rrf_merge_lists(
    lists: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Reciprocal-rank fusion over N ranked lists (cross-variant merge).

    RRF is rank-based and list-length-insensitive, which is exactly what is
    needed one level up from vector+BM25 fusion — merging the ranked outputs
    of multiple query variants into one candidate pool. Per-list ``weights``
    default to unit (equal contribution per variant).
    """
    if not lists:
        return []
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must have one entry per list")

    base: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for rank_list, weight in zip(lists, weights):
        for rank, chunk in enumerate(rank_list):
            base.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank + 1)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged: list[RetrievedChunk] = []
    for chunk_id in ordered:
        merged.append(
            base[chunk_id].model_copy(
                deep=True,
                update={
                    "fused_score": round(scores[chunk_id], 6),
                    "retrieval_metadata": {
                        **base[chunk_id].retrieval_metadata,
                        "rrf": {
                            "k": k,
                            "n_lists": len(lists),
                            "weights": list(weights),
                            "rrf_score": scores[chunk_id],
                        },
                    },
                },
            )
        )
    return merged


def weighted_merge_lists(
    lists: Sequence[Sequence[RetrievedChunk]],
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Weighted-score merge over N ranked lists (cross-variant merge).

    Each list's per-chunk score (``fused_score``) is min-max normalized to
    [0,1] before the weighted sum; per-list ``weights`` default to unit.
    """
    if not lists:
        return []
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must have one entry per list")

    base: dict[str, RetrievedChunk] = {}
    normalized: list[list[float]] = []
    for rank_list in lists:
        for chunk in rank_list:
            base.setdefault(chunk.chunk_id, chunk)
        normalized.append(_normalize_scores([c.fused_score for c in rank_list]))

    ids = {c.chunk_id for rank_list in lists for c in rank_list}
    fused: list[RetrievedChunk] = []
    for chunk_id in ids:
        score = 0.0
        for i, rank_list in enumerate(lists):
            for j, chunk in enumerate(rank_list):
                if chunk.chunk_id == chunk_id:
                    score += weights[i] * normalized[i][j]
                    break
        fused.append(
            base[chunk_id].model_copy(
                deep=True,
                update={
                    "fused_score": round(score, 6),
                    "retrieval_metadata": {
                        **base[chunk_id].retrieval_metadata,
                        "weighted_fusion": {
                            "n_lists": len(lists),
                            "weights": list(weights),
                        },
                    },
                },
            )
        )
    return sorted(fused, key=lambda c: (c.fused_score or 0.0), reverse=True)


def _normalize_scores(scores: Iterable[float | None]) -> list[float]:
    values = [s if s is not None else 0.0 for s in scores]
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


__all__ = ["weighted_fusion", "rrf_fusion", "weighted_merge_lists", "rrf_merge_lists"]