"""Unit tests for query rewriting: heuristics, LLM rewriter, rewrite service,
multi-query retrieval fan-out, and the RAG-service gate."""

from __future__ import annotations

import pytest

from app.application.services.context import ContextBuilder
from app.application.services.llm_service import LLMService
from app.application.services.rag import RAGService
from app.application.services.retrieval import RetrievalService
from app.application.services.rewrite import RewriteService
from app.domain.entities import RetrievedChunk
from app.domain.exceptions import LLMError, RetrievalError
from app.domain.interfaces import QueryRewriter
from app.domain.value_objects import MAX_REWRITE_VARIANTS, ConfigurableFusion, RewrittenQuery, SearchFilters
from app.infrastructure.llm.echo import EchoLLM
from app.infrastructure.rewriters.heuristic import HeuristicQueryRewriter
from app.infrastructure.rewriters.llm import LLMQueryRewriter
from tests.fakes import FakeCache


# --- Test doubles -----------------------------------------------------------


class StubLLM:
    """LLM-protocol fake with an injectable raw response or backend failure."""

    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "stub"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.fail:
            raise LLMError("backend unavailable")
        return self.text


def make_llm_service(stub: StubLLM) -> LLMService:
    return LLMService(
        llm=stub,
        cache=None,
        default_temperature=0.0,
        default_max_tokens=256,
        providers={"nemotron": stub},
        default_model="nemotron",
    )


class RecordingQueryRewriter:
    """Records args and either returns variants or raises (injectable)."""

    def __init__(self, strategy: str = "llm", fail: bool = False) -> None:
        self.strategy = strategy
        self.fail = fail
        self.calls: list[tuple] = []

    def rewrite(self, query: str, *, max_variants: int, context_hint: str | None = None) -> RewrittenQuery:
        self.calls.append((query, max_variants, context_hint))
        if self.fail:
            raise LLMError("rewriter failure")
        return RewrittenQuery(original=query, variants=[query, f"{query} variant"], strategy=self.strategy)


class RecordingHybrid:
    """HybridSearch double recording every fan-out call (query/top_k/filters)."""

    def __init__(self, empty: bool = False) -> None:
        self.calls: list[dict] = []
        self.empty = empty

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        semantic_top_k: int | None = None,
        lexical_top_k: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "query_id": query_id}
        )
        if self.empty:
            return []
        return [
            RetrievedChunk(
                chunk_id=f"chunk:{query}",
                document_id="doc_1",
                filename="f.txt",
                text=f"text for {query}",
            )
        ]


class RecordingRetrieval:
    """RetrievalService-level double recording single vs multi-query paths."""

    def __init__(self) -> None:
        self.retrieve_calls: list[tuple] = []
        self.multi_calls: list[tuple] = []

    def retrieve(self, query: str, top_k: int | None = None, *, filters=None, query_id: str | None = None):
        self.retrieve_calls.append((query, top_k, filters, query_id))
        return []

    def retrieve_multi(self, rewritten, top_k: int | None = None, *, filters=None, query_id: str | None = None):
        self.multi_calls.append((rewritten, top_k, filters, query_id))
        return []


class FailingQueryRewriter:
    def rewrite(self, query: str, *, max_variants: int, context_hint: str | None = None) -> RewrittenQuery:
        raise RuntimeError("everything is on fire")


# --- HeuristicQueryRewriter -------------------------------------------------


class TestHeuristicQueryRewriter:
    def test_original_always_first(self) -> None:
        result = HeuristicQueryRewriter().rewrite("the refund policy", max_variants=3)
        assert result.original == "the refund policy"
        assert result.variants[0] == "the refund policy"
        assert result.strategy == "heuristic"

    def test_normalizes_typos_and_contractions(self) -> None:
        result = HeuristicQueryRewriter().rewrite("hte recieve refund", max_variants=5)
        assert "receive refund" in result.variants

    def test_expands_synonyms(self) -> None:
        result = HeuristicQueryRewriter().rewrite("refund policy", max_variants=5)
        assert "refunds policy" in result.variants
        assert "refund policies" in result.variants

    def test_deterministic(self) -> None:
        a = HeuristicQueryRewriter().rewrite("what is the return policy", max_variants=4)
        b = HeuristicQueryRewriter().rewrite("what is the return policy", max_variants=4)
        assert a.variants == b.variants

    def test_caps_variants(self) -> None:
        result = HeuristicQueryRewriter().rewrite("can I get a refund for shipping costs", max_variants=1000)
        assert len(result.variants) <= MAX_REWRITE_VARIANTS

    def test_does_not_duplicate_when_normalization_is_a_noop(self) -> None:
        result = HeuristicQueryRewriter().rewrite("acme handbook", max_variants=3)
        assert result.variants == ["acme handbook"]

    def test_never_raises_on_empty(self) -> None:
        result = HeuristicQueryRewriter().rewrite("   ", max_variants=3)
        assert result.original == ""
        assert result.variants == [""]


# --- LLMQueryRewriter -------------------------------------------------------


class TestLLMQueryRewriter:
    def test_prompt_builds_and_passes_params(self) -> None:
        stub = StubLLM(text="variant one\n2. variant two\n- variant three\n")
        rewriter = LLMQueryRewriter(
            make_llm_service(stub), timeout_seconds=1.5, max_tokens=128
        )
        result = rewriter.rewrite("original query", max_variants=3)
        assert stub.calls
        call = stub.calls[0]
        assert "original query" in call["prompt"]
        assert call["system_prompt"]
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 128
        assert call["timeout_seconds"] == 1.5
        assert result.variants == ["original query", "variant one", "variant two"]

    def test_strips_numbering_quotes_and_dedupes(self) -> None:
        stub = StubLLM(text='1. "dup"\n2. dup\n3. different\n')
        result = LLMQueryRewriter(make_llm_service(stub)).rewrite("q", max_variants=3)
        assert result.variants == ["q", "dup", "different"]

    def test_truncates_long_variants(self) -> None:
        stub = StubLLM(text=f"{'x' * 500}\n")
        rewriter = LLMQueryRewriter(make_llm_service(stub), max_variant_length=200)
        result = rewriter.rewrite("q", max_variants=1)
        assert len(result.variants[1]) <= 200

    def test_empty_response_degrades_to_original(self) -> None:
        stub = StubLLM(text="")
        result = LLMQueryRewriter(make_llm_service(stub)).rewrite("q", max_variants=5)
        assert result.variants == ["q"]

    def test_caps_variants(self) -> None:
        stub = StubLLM(text="\n".join(f"v{i}" for i in range(50)))
        result = LLMQueryRewriter(make_llm_service(stub)).rewrite("q", max_variants=60)
        assert len(result.variants) <= MAX_REWRITE_VARIANTS

    def test_backend_failure_propagates(self) -> None:
        stub = StubLLM(fail=True)
        rewriter = LLMQueryRewriter(make_llm_service(stub))
        with pytest.raises(LLMError):
            rewriter.rewrite("q", max_variants=3)


# --- RewriteService ---------------------------------------------------------


class TestRewriteService:
    def test_default_strategy_calls_llm_rewriter(self) -> None:
        recorder = RecordingQueryRewriter()
        service = RewriteService(recorder, HeuristicQueryRewriter(), default_strategy="llm")
        service.rewrite("q")
        assert recorder.calls and recorder.calls[0][0] == "q"

    def test_strategy_override(self) -> None:
        recorder = RecordingQueryRewriter()
        service = RewriteService(recorder, HeuristicQueryRewriter(), default_strategy="llm")
        result = service.rewrite("q", strategy="heuristic")
        assert not recorder.calls
        assert result.strategy == "heuristic"

    def test_invalid_strategy_rejected(self) -> None:
        service = RewriteService(RecordingQueryRewriter(), HeuristicQueryRewriter())
        with pytest.raises(ValueError):
            service.rewrite("q", strategy="bogus")

    def test_max_variants_clamped_to_cap(self) -> None:
        recorder = RecordingQueryRewriter()
        service = RewriteService(recorder, HeuristicQueryRewriter())
        service.rewrite("q", max_variants=999)
        assert recorder.calls[0][1] == MAX_REWRITE_VARIANTS

    def test_llm_failure_falls_back_to_heuristic(self) -> None:
        failing = RecordingQueryRewriter(fail=True)
        service = RewriteService(failing, HeuristicQueryRewriter())
        result = service.rewrite("q", max_variants=3)
        assert result.strategy == "llm_fallback_heuristic"
        assert result.variants[0] == "q"

    def test_llm_rewriter_none_falls_back_fast(self) -> None:
        service = RewriteService(None, HeuristicQueryRewriter(), default_strategy="llm")
        result = service.rewrite("q")
        assert result.strategy == "heuristic"

    def test_heuristic_failure_degrades_to_passthrough(self) -> None:
        service = RewriteService(RecordingQueryRewriter(fail=True), FailingQueryRewriter())
        result = service.rewrite("q")
        assert result.strategy == "passthrough"
        assert result.variants == ["q"]

    def test_cache_hit_avoids_llm(self) -> None:
        recorder = RecordingQueryRewriter()
        cache = FakeCache()
        service = RewriteService(recorder, HeuristicQueryRewriter(), cache=cache)
        service.rewrite("q", max_variants=3)
        service.rewrite("q", max_variants=3)
        assert len(recorder.calls) == 1

    def test_cache_key_separates_strategy_and_count(self) -> None:
        recorder = RecordingQueryRewriter()
        cache = FakeCache()
        service = RewriteService(recorder, HeuristicQueryRewriter(), cache=cache)
        service.rewrite("q", max_variants=3)
        service.rewrite("q", max_variants=1)  # different count -> different key
        service.rewrite("q", max_variants=3, strategy="heuristic")  # different strategy
        assert len(recorder.calls) == 2

    def test_context_hint_does_not_affect_cache_key(self) -> None:
        recorder = RecordingQueryRewriter()
        cache = FakeCache()
        service = RewriteService(recorder, HeuristicQueryRewriter(), cache=cache)
        service.rewrite("q", max_variants=3, context_hint="previous turns")
        service.rewrite("q", max_variants=3, context_hint="something else")
        assert len(recorder.calls) == 1

    def test_broken_cache_does_not_break_rewriting(self) -> None:
        recorder = RecordingQueryRewriter()
        service = RewriteService(recorder, HeuristicQueryRewriter(), cache=FakeCache(fail=True))
        assert service.rewrite("q").variants[0] == "q"


# --- RetrievalService.retrieve_multi -----------------------------------------


class TestRetrieveMulti:
    def _service(self, mode: str = "rrf") -> RetrievalService:
        return RetrievalService(
            RecordingHybrid(),
            reranker=None,
            fusion=ConfigurableFusion(mode=mode, rrf_k=60, weight_semantic=0.5),
        )

    def test_fan_out_one_call_per_variant_same_filters(self) -> None:
        hybrid = RecordingHybrid()
        service = RetrievalService(hybrid, reranker=None)
        filters = SearchFilters(document_ids=["doc_1", "doc_2"])
        rewritten = RewrittenQuery(original="q", variants=["q", "q two", "q three"])
        service.retrieve_multi(rewritten, top_k=5, filters=filters, query_id="abc")
        assert [c["query"] for c in hybrid.calls] == ["q", "q two", "q three"]
        assert all(c["top_k"] == 5 for c in hybrid.calls)
        assert all(c["query_id"] == "abc" for c in hybrid.calls)
        # Access control must reach every variant identically.
        for call in hybrid.calls:
            assert call["filters"] is filters

    def test_merges_variants_with_rrf(self) -> None:
        service = self._service(mode="rrf")
        rewritten = RewrittenQuery(original="q", variants=["q", "q two"])
        results = service.retrieve_multi(rewritten, top_k=5)
        assert {c.chunk_id for c in results} == {"chunk:q", "chunk:q two"}
        assert all(c.retrieval_metadata.get("rrf") for c in results)

    def test_merges_variants_with_weighted_fusion(self) -> None:
        service = self._service(mode="weighted")
        rewritten = RewrittenQuery(original="q", variants=["q", "q two"])
        results = service.retrieve_multi(rewritten, top_k=5)
        assert {c.chunk_id for c in results} == {"chunk:q", "chunk:q two"}

    def test_single_variant_is_harmless(self) -> None:
        service = self._service(mode="rrf")
        rewritten = RewrittenQuery(original="q", variants=["q"])
        results = service.retrieve_multi(rewritten, top_k=5)
        assert [c.chunk_id for c in results] == ["chunk:q"]

    def test_empty_query_rejected(self) -> None:
        service = self._service()
        with pytest.raises(RetrievalError):
            service.retrieve_multi(RewrittenQuery(original="", variants=[""]), top_k=5)

    def test_no_candidates_anywhere_returns_empty(self) -> None:
        hybrid = RecordingHybrid(empty=True)
        service = RetrievalService(hybrid, reranker=None)
        results = service.retrieve_multi(RewrittenQuery(original="q", variants=["q"]), top_k=5)
        assert results == []

    def test_dedupe_across_variants(self) -> None:
        hybrid = RecordingHybrid()
        service = RetrievalService(hybrid, reranker=None)
        rewritten = RewrittenQuery(original="same", variants=["same", "same"])
        results = service.retrieve_multi(rewritten, top_k=5)
        assert len(results) == 1


# --- RAGService rewrite gate -------------------------------------------------


class TestRAGServiceGate:
    def _rag(self, enabled: bool) -> RAGService:
        retrieval = RecordingRetrieval()
        rewrite_service = (
            RewriteService(HeuristicQueryRewriter(), HeuristicQueryRewriter(), default_strategy="heuristic")
            if enabled
            else None
        )
        llm_service = LLMService(EchoLLM(), cache=None)
        rag = RAGService(
            retrieval,
            ContextBuilder(8000),
            llm_service,
            rewrite_service=rewrite_service,
        )
        return rag

    def test_disabled_uses_single_query_path(self) -> None:
        rag = self._rag(enabled=False)
        response = rag.answer("what is the policy", top_k=3)
        retrieval: RecordingRetrieval = rag.retrieval
        assert len(retrieval.retrieve_calls) == 1
        assert not retrieval.multi_calls
        assert "rewritten_variants" not in response.metadata
        assert "rewrite_strategy" not in response.metadata

    def test_disabled_ignores_rewrite_true_override(self) -> None:
        rag = self._rag(enabled=False)
        response = rag.answer("q", top_k=3, rewrite=True)
        retrieval: RecordingRetrieval = rag.retrieval
        assert len(retrieval.retrieve_calls) == 1
        assert not retrieval.multi_calls
        assert "rewritten_variants" not in response.metadata

    def test_enabled_uses_multi_query_path(self) -> None:
        rag = self._rag(enabled=True)
        response = rag.answer("what is the refund policy", top_k=3)
        retrieval: RecordingRetrieval = rag.retrieval
        assert not retrieval.retrieve_calls
        assert len(retrieval.multi_calls) == 1
        assert response.metadata["rewritten_variants"]
        assert response.metadata["rewrite_strategy"] == "heuristic"

    def test_enabled_respects_false_override(self) -> None:
        rag = self._rag(enabled=True)
        rag.answer("q", top_k=3, rewrite=False)
        retrieval: RecordingRetrieval = rag.retrieval
        assert len(retrieval.retrieve_calls) == 1
        assert not retrieval.multi_calls

    def test_access_filters_reach_retrieve_multi(self) -> None:
        rag = self._rag(enabled=True)
        rag.answer("q", top_k=3, accessible_document_ids={"doc_42"})
        retrieval: RecordingRetrieval = rag.retrieval
        rewritten, _top_k, filters, _qid = retrieval.multi_calls[0]
        assert rewritten.variants
        assert filters is not None
        assert filters.document_ids == ["doc_42"]