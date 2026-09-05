"""RAGService response scoping to an allowed set of document ids."""

from __future__ import annotations

from app.application.services.context import ContextBuilder
from app.application.services.llm_service import LLMService
from app.application.services.rag import RAGService
from app.domain.entities import RetrievedChunk
from app.domain.value_objects import SearchFilters
from app.infrastructure.llm.echo import EchoLLM
from tests.fakes import NoopReranker, make_chunk  # noqa: F401


class StubRetrieval:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_filters = None

    def retrieve(self, query: str, **kwargs):
        self.last_filters = kwargs.get("filters")
        return [c.model_copy() for c in self.chunks]


def _chunk(document_id: str) -> RetrievedChunk:
    chunk = make_chunk(document_id=document_id, text=f"content of {document_id}")
    return RetrievedChunk.from_chunk(chunk, fused_score=1.0)


def _service(chunks: list[RetrievedChunk]) -> RAGService:
    llm_service = LLMService(EchoLLM())
    return RAGService(StubRetrieval(chunks), ContextBuilder(8000), llm_service)


def test_no_scoping_returns_all() -> None:
    rag = _service([_chunk("doc_a"), _chunk("doc_b")])
    response = rag.answer("query", top_k=10)
    assert sorted(c.document_id for c in response.retrieved_chunks) == ["doc_a", "doc_b"]


def test_scoping_filters_out_chunks() -> None:
    rag = _service([_chunk("doc_a"), _chunk("doc_b"), _chunk("doc_c")])
    response = rag.answer("query", top_k=10, accessible_document_ids={"doc_a", "doc_c"})
    assert sorted(c.document_id for c in response.retrieved_chunks) == ["doc_a", "doc_c"]


def test_scoping_to_empty_yields_no_context() -> None:
    rag = _service([_chunk("doc_a")])
    response = rag.answer("query", top_k=10, accessible_document_ids=set())
    assert response.retrieved_chunks == []
    assert response.citations == []


def test_scoping_is_pushed_down_as_search_filters() -> None:
    retrieval = StubRetrieval([_chunk("doc_a"), _chunk("doc_b")])
    rag = RAGService(retrieval, ContextBuilder(8000), LLMService(EchoLLM()))
    rag.answer("query", top_k=10, accessible_document_ids={"doc_a"})
    assert retrieval.last_filters is not None
    assert retrieval.last_filters.document_ids == ["doc_a"]


def test_both_filters_and_scoping_merged() -> None:
    retrieval = StubRetrieval([_chunk("doc_a")])
    rag = RAGService(retrieval, ContextBuilder(8000), LLMService(EchoLLM()))
    rag.answer("query", top_k=10, filters=SearchFilters(exact={"department": "finance"}), accessible_document_ids={"doc_a"})
    assert retrieval.last_filters.document_ids == ["doc_a"]
    assert retrieval.last_filters.exact == {"department": "finance"}


def test_retrieval_cannot_leak_restricted_chunks() -> None:
    # Defense-in-depth: even if a buggy retriever ignores the filters, the
    # post-filter safety net must drop documents outside the allowed set.
    retrieval = StubRetrieval([_chunk("doc_a"), _chunk("doc_b"), _chunk("doc_c")])
    rag = RAGService(retrieval, ContextBuilder(8000), LLMService(EchoLLM()))
    response = rag.answer("query", top_k=10, accessible_document_ids={"doc_b"})
    assert [c.document_id for c in response.retrieved_chunks] == ["doc_b"]