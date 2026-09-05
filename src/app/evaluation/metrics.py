"""Retrieval + generation evaluation metrics (no external evaluation platform).

Retrieval: Recall@K, Precision@K, MRR, nDCG — all computable from a ranked list
of chunk ids against the set of "relevant" chunk ids in an evaluation sample.

Generation: hooks for faithfulness / answer relevance / citation correctness.
Faithfulness here uses token-overlap between the answer and the grounding
context (a lightweight, deterministic approximation of faithfulness /
claim-level grounding — see docs/evaluation.md for caveats).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

_WORD_PATTERN = re.compile(r"\w+")


def _token_set(text: str) -> set[str]:
    return set(t for t in _WORD_PATTERN.findall(text.lower()) if len(t) > 2)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / min(k, len(ranked_ids)) if ranked_ids else 0.0


def mrr(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked_ids):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    k = min(k, len(ranked_ids))
    if k == 0:
        return 0.0

    dcg = 0.0
    for i in range(k):
        if ranked_ids[i] in relevant:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: all |relevant| relevant items would sit at the top.
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def evaluate_retrieval(
    ranked_lists: Iterable[Sequence[str]],
    relevant_sets: Iterable[set[str]],
    k_values: Iterable[int],
) -> dict[str, list[float]]:
    """Aggregate retrieval metrics over many queries.

    Returns mean metrics keyed like ``recall@5`` for each requested k plus
    ``mrr`` and ``ndcg@k`` for each k.
    """
    k_values = list(k_values)
    acc: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    acc.update({f"precision@{k}": [] for k in k_values})
    acc.update({f"ndcg@{k}": [] for k in k_values})
    acc["mrr"] = []

    for ranked, relevant in zip(ranked_lists, relevant_sets):
        relevant = set(relevant)
        for k in k_values:
            acc[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            acc[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            acc[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
        acc["mrr"].append(mrr(ranked, relevant))

    return {k: _mean(v) for k, v in acc.items()}


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def faithfulness(answer: str, evidence: str) -> float:
    """Token-overlap faithfulness: fraction of answer content tokens that appear
    in the evidence/context.

    NOTE: This is a lightweight, *heuristic* proxy. It rewards answer tokens
    that also appear in the context and punishes hallucinated tokens. It is not
    a substitute for claim-level NLI-based faithfulness scoring.
    """
    answer_tokens = _token_set(answer)
    evidence_tokens = _token_set(evidence)
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def answer_relevance(answer: str, question: str) -> float:
    """Token-overlap relevance between answer and question (lexical proxy)."""
    overlap = _token_set(answer) & _token_set(question)
    union = _token_set(answer) | _token_set(question)
    if not union:
        return 0.0
    return round(len(overlap) / len(union), 4)


def context_relevance(retrieved_chunks: Sequence[str], relevant: set[str]) -> float:
    """Fraction of retrieved chunks that are relevant (i.e. Precision@|k|)."""
    if not retrieved_chunks:
        return 0.0
    hit = sum(1 for cid in retrieved_chunks if cid in relevant)
    return round(hit / len(retrieved_chunks), 4)


def citation_correctness(cited_chunk_ids: Sequence[str], relevant: set[str]) -> float:
    """Fraction of cited chunks that are genuinely relevant."""
    if not cited_chunk_ids:
        return 0.0
    hit = sum(1 for cid in cited_chunk_ids if cid in relevant)
    return round(hit / len(cited_chunk_ids), 4)


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "evaluate_retrieval",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "citation_correctness",
]