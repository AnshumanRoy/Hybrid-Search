"""BM25 lexical retriever tests."""

import pytest

from app.domain.value_objects import SearchFilters
from app.infrastructure.retrievers.bm25 import BM25Retriever, tokenize
from tests.fakes import make_chunk


def test_retrieve_lexical_relevance() -> None:
    retriever = BM25Retriever()
    retriever.add_documents(
        [
            make_chunk(chunk_id="c1", text="The refund policy allows returns within 30 days."),
            make_chunk(chunk_id="c2", text="Apples are a delicious fruit enjoyed worldwide."),
            make_chunk(chunk_id="c3", text="Shipping policy covers delivery times and fees."),
        ]
    )
    results = retriever.retrieve("refund policy returns", top_k=3)
    assert results[0].chunk_id == "c1"
    assert results[0].lexical_score is not None
    assert results[0].lexical_score > 0


def test_retrieve_top_k_bound() -> None:
    retriever = BM25Retriever()
    retriever.add_documents([make_chunk(chunk_id=f"c{i}", text=f"unique token {i}" * 5) for i in range(10)])
    results = retriever.retrieve("token", top_k=3)
    assert len(results) == 3


def test_empty_corpus_returns_empty() -> None:
    assert BM25Retriever().retrieve("anything", top_k=5) == []


def test_remove_documents_changes_results() -> None:
    retriever = BM25Retriever()
    retriever.add_documents(
        [
            make_chunk(chunk_id="keep", text="refund policy details here"),
            make_chunk(chunk_id="drop", text="refund policy unrelated chatter"),
        ]
    )
    retriever.remove_documents(["drop"])
    results = retriever.retrieve("refund", top_k=5)
    assert [c.chunk_id for c in results] == ["keep"]
    assert retriever.count() == 1


def test_count() -> None:
    retriever = BM25Retriever()
    retriever.add_documents([make_chunk(chunk_id="a"), make_chunk(chunk_id="b")])
    assert retriever.count() == 2


def test_tokenize_lowercases_and_strips_punct() -> None:
    assert tokenize("Refund, policy!") == ["refund", "policy"]


def test_retrieve_filters_by_document_ids() -> None:
    retriever = BM25Retriever()
    retriever.add_documents(
        [
            make_chunk(document_id="doc_b", chunk_id="b1", text="refund policy refund policy refund policy refund policy"),
            make_chunk(document_id="doc_a", chunk_id="a1", text="refund policy allows returns within 30 days"),
            make_chunk(document_id="doc_a", chunk_id="a2", text="cancellation refunded to original payment method"),
        ]
    )

    # Unfiltered: the doc_b chunk ranks first (repeated query terms).
    unfiltered = retriever.retrieve("refund policy", top_k=5)
    assert unfiltered[0].chunk_id == "b1"

    # Filtered: disallowed chunks are skipped before the top_k cut, so the best
    # allowed chunk is still found even when it ranks below restricted content.
    allowed = retriever.retrieve("refund policy", top_k=1, filters=SearchFilters(document_ids=["doc_a"]))
    assert len(allowed) == 1
    assert allowed[0].chunk_id == "a1"
    assert allowed[0].document_id == "doc_a"


def test_relevance_token_query_returns_zero() -> None:
    retriever = BM25Retriever()
    retriever.add_documents([make_chunk(chunk_id="c1", text="completely different topics here")])
    results = retriever.retrieve("zzz", top_k=5)
    assert results == [] or all(c.lexical_score == 0 for c in results)