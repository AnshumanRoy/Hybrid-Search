"""Evaluation metric tests with hand-computable ground truth."""

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


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 5) == 1.0


def test_precision_at_k() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 5) == 0.4


def test_mrr() -> None:
    assert mrr(["a", "b", "c"], {"c"}) == 1 / 3
    assert mrr(["a", "b"], {"c"}) == 0.0
    assert mrr(["a"], {"a"}) == 1.0


def test_ndcg_at_k() -> None:
    # All relevant at the top => ndcg 1.0
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, {"a", "b"}, 2) > 0.99
    # Nothing relevant => 0
    assert ndcg_at_k(ranked, {"zzz"}, 2) == 0.0


def test_evaluate_retrieval_aggregates() -> None:
    report = evaluate_retrieval([["a", "b"], ["a", "b"]], [{"a"}, {"b"}], k_values=[1, 2])
    assert report["recall@1"] == 0.5  # (1.0 + 0.0) / 2
    assert report["recall@2"] == 1.0
    assert 0.0 <= report["mrr"] <= 1.0


def test_faithfulness_lexical_proxy() -> None:
    answer = "The refund policy allows returns within thirty days."
    evidence = "The refund policy allows returns within thirty days after purchase."
    assert faithfulness(answer, evidence) > 0.5
    assert faithfulness("Totally unrelated fantasy text.", evidence) <= 0.5


def test_answer_relevance() -> None:
    assert answer_relevance("What is the refund window?", "The refund window") > 0.0


def test_context_relevance() -> None:
    assert context_relevance(["good", "bad"], {"good"}) == 0.5
    assert context_relevance([], {"good"}) == 0.0


def test_citation_correctness() -> None:
    assert citation_correctness(["a", "b"], {"a", "c"}) == 0.5
    assert citation_correctness([], {"a"}) == 0.0