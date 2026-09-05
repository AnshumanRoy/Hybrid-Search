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
from app.domain.entities import RAGResponse
from app.domain.exceptions import LLMError
from app.domain.value_objects import SearchFilters

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        retrieval: RetrievalService,
        context_builder: ContextBuilder,
        llm: LLMService,
        cacheable_generation: bool = True,
    ) -> None:
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.llm = llm
        self.cacheable_generation = cacheable_generation

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        max_context_chars: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
        accessible_document_ids: set[str] | None = None,
    ) -> RAGResponse:
        qid = query_id or uuid.uuid4().hex
        overall_started = time.perf_counter()

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

        retrieval_started = time.perf_counter()
        chunks = self.retrieval.retrieve(query, top_k=top_k, filters=retrieval_filters, query_id=qid)
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

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            metadata={
                "query_id": qid,
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
                "retrieval_count": len(chunks),
                "context_chars": len(context),
                "llm_model": self.llm.llm.model_name,
            },
        )