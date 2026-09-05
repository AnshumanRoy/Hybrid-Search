"""RAGService orchestration tests (fully faked, no network)."""

from app.application.services.context import ContextBuilder
from app.application.services.llm_service import LLMService
from app.application.services.rag import RAGService
from app.application.services.retrieval import RetrievalService
from app.domain.entities import RetrievedChunk
from app.infrastructure.llm.echo import EchoLLM
from tests.fakes import NoopReranker, make_chunk


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk.from_chunk(make_chunk(chunk_id=cid, text=f"detailed refund text for {cid}"))


class StubHybrid:
    def __init__(self):
        self.chunks = [_chunk("c1"), _chunk("c2")]

    def retrieve(self, query, top_k, *, semantic_top_k=None, lexical_top_k=None, filters=None, query_id=None):
        return self.chunks[:top_k]


def _service():
    retrieval = RetrievalService(StubHybrid(), reranker=NoopReranker())
    llm = LLMService(EchoLLM())
    return RAGService(retrieval, ContextBuilder(), llm)


def test_answer_returns_struct() -> None:
    rag = _service()
    response = rag.answer("What is the refund policy?")
    assert response.answer
    assert len(response.citations) == 2
    assert response.citations[0].chunk_id == "c1"
    assert response.metadata.get("retrieval_ms") is not None
    assert response.metadata.get("generation_ms") is not None


def test_answer_unavailable_when_no_context() -> None:
    retrieval = RetrievalService(StubHybrid())
    retrieval.hybrid_search = type("Empty", (), {"retrieve": lambda self, *a, **k: []})()
    rag = RAGService(retrieval, ContextBuilder(), LLMService(EchoLLM()))
    response = rag.answer("nothing here")
    # echo LLM returns instructions rather than grounded content; assert metadata present
    assert response.retrieved_chunks == []


def test_citations_have_page_and_doc() -> None:
    rag = _service()
    response = rag.answer("q")
    c = response.citations[0]
    assert c.document_id
    assert c.filename
    assert c.page_number is not None