"""Model selection at the RAG layer: backend swap must not change context."""

from __future__ import annotations

import pytest

from app.application.services.context import ContextBuilder
from app.application.services.llm_service import LLMService
from app.application.services.rag import RAGService
from app.domain.entities import RetrievedChunk
from app.domain.exceptions import UnknownModelError
from app.infrastructure.llm.echo import EchoLLM
from tests.fakes import RecordingLLM, make_chunk


class StubRetrieval:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, query: str, **kwargs):
        self.calls += 1
        return [c.model_copy() for c in self.chunks]


def _chunk(document_id: str) -> RetrievedChunk:
    return RetrievedChunk.from_chunk(make_chunk(document_id=document_id, text=f"content of {document_id}"), fused_score=1.0)


def _rag_service(chunks: list[RetrievedChunk]) -> RAGService:
    echo = RecordingLLM("echo")
    nemotron = RecordingLLM("nemotron")
    llm = LLMService(
        llm=echo,
        providers={"echo": echo, "nemotron": nemotron},
        default_model="echo",
    )
    return RAGService(StubRetrieval(chunks), ContextBuilder(8000), llm)


def test_unknown_model_fails_before_retrieval() -> None:
    retrieval = StubRetrieval([_chunk("doc_a")])
    llm = LLMService(
        llm=RecordingLLM("echo"),
        providers={"echo": RecordingLLM("echo"), "nemotron": RecordingLLM("nemotron")},
        default_model="echo",
    )
    rag = RAGService(retrieval, ContextBuilder(8000), llm)
    with pytest.raises(UnknownModelError, match="unknown model: bogus"):
        rag.answer("query", top_k=10, model="bogus")
    assert retrieval.calls == 0


def test_prompt_identical_across_models() -> None:
    echo = RecordingLLM("echo")
    nemotron = RecordingLLM("nemotron")
    llm = LLMService(
        llm=echo,
        providers={"echo": echo, "nemotron": nemotron},
        default_model="echo",
    )
    retrieval = StubRetrieval([_chunk("doc_a"), _chunk("doc_b")])
    rag = RAGService(retrieval, ContextBuilder(8000), llm)

    rag.answer("What is the refund policy?", top_k=10, model="echo", accessible_document_ids={"doc_a"})
    rag.answer("What is the refund policy?", top_k=10, model="nemotron", accessible_document_ids={"doc_a"})

    echo_prompt = echo.calls[0][0]
    nemotron_prompt = nemotron.calls[0][0]
    assert echo_prompt == nemotron_prompt
    assert _chunk_ids_in_prompt(echo_prompt) == ["doc_a.txt"]
    assert "doc_b" not in echo_prompt


def test_default_model_recorded_in_metadata() -> None:
    echo = RecordingLLM("echo")
    nemotron = RecordingLLM("nemotron")
    llm = LLMService(
        llm=echo,
        providers={"echo": echo, "nemotron": nemotron},
        default_model="echo",
    )
    rag = RAGService(StubRetrieval([_chunk("doc_a")]), ContextBuilder(8000), llm)

    assert rag.answer("q", top_k=10).metadata["llm_model"] == "echo"
    assert rag.answer("q", top_k=10, model="nemotron").metadata["llm_model"] == "nemotron"


def _chunk_ids_in_prompt(prompt: str) -> list[str]:
    import re

    return re.findall(r"Document:\s*([\w.-]+)", prompt)


def test_default_backend_works_without_any_model_param() -> None:
    rag = _rag_service([_chunk("doc_a")])
    response = rag.answer("q", top_k=10)
    assert response.metadata["llm_model"] == "echo"
    assert response.answer