"""Evaluation module (retrieval + generation metrics, runner, script)."""

from app.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    evaluate_retrieval,
    faithfulness,
    answer_relevance,
    context_relevance,
    citation_correctness,
)
from app.evaluation.runner import EvaluationRunner

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
    "EvaluationRunner",
]