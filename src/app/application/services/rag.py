"""High-level RAG orchestration: retrieval → context → prompt → LLM → response.

``RAGService`` orchestrates collaborators; it implements no internals itself.
"""

from __future__ import annotations

import logging
import time
import uuid

from app.application.services.context import ContextBuilder
from app.application.services.prompt import build_system_prompt, build_user_prompt, citations_from_chunks
from app.application.services.retrieval import RetrievalService
from app.application.services.llm_service import LLMService
from app.application.services.rewrite import RewriteService
from app.domain.entities import RAGResponse
from app.domain.exceptions import LLMError
from app.domain.value_objects import RewrittenQuery, SearchFilters

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        retrieval: RetrievalService,
        context_builder: ContextBuilder,
        llm: LLMService,
        cacheable_generation: bool = True,
        rewrite_service: RewriteService | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.llm = llm
        self.cacheable_generation = cacheable_generation
        self.rewrite_service = rewrite_service

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        max_context_chars: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
        accessible_document_ids: set[str] | None = None,
        model: str | None = None,
        rewrite: bool | None = None,
        max_variants: int | None = None,
    ) -> RAGResponse:
        qid = query_id or uuid.uuid4().hex
        overall_started = time.perf_counter()

        # Resolve the model before spending any time on retrieval — an unknown
        # model name must fail fast and cheaply (raise UnknownModelError).
        model_name = self.llm.model_name_for(model)

        # Access control is pushed down into retrieval (both vector + lexical
        # indexes are filtered before fusion/reranking); the post-filter below
        # remains as a defense-in-depth safety net against a buggy store filter.
        retrieval_filters = filters
        if accessible_document_ids is not None:
            allowed = list(accessible_document_ids)
            if retrieval_filters is None:
                retrieval_filters = SearchFilters(document_ids=allowed)
            else:
                retrieval_filters = retrieval_filters.model_copy(update={"document_ids": allowed})

        # Query rewriting gate: the container injects rewrite_service only when
        # QUERY_REWRITE_ENABLED=true. Request-level rewrite overrides default to
        # the enabled state; a rewrite=true override while the feature is
        # disabled (service is None) silently degrades to single-query.
        rewritten: RewrittenQuery | None = None
        if self.rewrite_service is not None and rewrite is not False:
            rewritten = self.rewrite_service.rewrite(query, max_variants=max_variants)
            query_for_retrieval: str | list[str] = rewritten.variants
        else:
            query_for_retrieval = query

        retrieval_started = time.perf_counter()
        if rewritten is not None:
            chunks = self.retrieval.retrieve_multi(
                rewritten, top_k=top_k, filters=retrieval_filters, query_id=qid
            )
        else:
            chunks = self.retrieval.retrieve(query_for_retrieval, top_k=top_k, filters=retrieval_filters, query_id=qid)
        if accessible_document_ids is not None:
            chunks = [c for c in chunks if c.document_id in accessible_document_ids]
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        context = self.context_builder.build(chunks, max_chars=max_context_chars)
        user_prompt = build_user_prompt(context, query)
        system_prompt = build_system_prompt()

        generation_started = time.perf_counter()
        try:
            answer = self.llm.generate(
                user_prompt,
                system_prompt=system_prompt,
                cacheable=self.cacheable_generation,
                model=model,
            )
        except LLMError:
            logger.exception("LLM generation failed; query_id=%s", qid)
            raise
        generation_ms = (time.perf_counter() - generation_started) * 1000

        citations = citations_from_chunks(chunks)
        total_ms = (time.perf_counter() - overall_started) * 1000

        logger.info(
            "rag answered; query_id=%s chunks=%d retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f",
            qid,
            len(chunks),
            retrieval_ms,
            generation_ms,
            total_ms,
        )

        metadata: dict[str, object] = {
            "query_id": qid,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
            "retrieval_count": len(chunks),
            "context_chars": len(context),
            "llm_model": model_name,
        }
        if rewritten is not None:
            metadata["rewritten_variants"] = rewritten.variants
            metadata["rewrite_strategy"] = rewritten.strategy

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            metadata=metadata,
        )