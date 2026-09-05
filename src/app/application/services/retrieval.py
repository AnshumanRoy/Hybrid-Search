"""Retrieval orchestration service: query → hybrid search → rerank → top N."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Protocol

from app.domain.entities import RetrievedChunk
from app.domain.exceptions import RetrievalError
from app.domain.value_objects import SearchFilters
from app.domain.interfaces import Reranker

logger = logging.getLogger(__name__)


class HybridSearch(Protocol):
    """Fuses semantic + lexical retrieval into one ranked candidate list."""

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        semantic_top_k: int | None = None,
        lexical_top_k: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


class RetrievalService:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: Reranker | None = None,
        default_top_k: int = 10,
        rerank_top_k: int | None = None,
        max_query_length: int = 1000,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.rerank_top_k = rerank_top_k
        self.max_query_length = max_query_length

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.default_top_k
        if self.reranker is not None and top_k == 1:
            top_k = self.rerank_top_k or top_k

        query = (query or "").strip()
        if not query:
            raise RetrievalError("query must not be empty")
        if len(query) > self.max_query_length:
            raise RetrievalError(f"query exceeds maximum length of {self.max_query_length} characters")

        qid = query_id or uuid.uuid4().hex
        started = time.perf_counter()

        candidates = self.hybrid_search.retrieve(query, top_k, filters=filters, query_id=qid)
        if not candidates:
            logger.info("retrieval returned no candidates; query_id=%s", qid)
            return []

        if self.reranker is not None:
            rerank_limit = self.rerank_top_k or max(top_k, len(candidates))
            candidates = [c for c in candidates if c.text.strip()]
            if candidates:
                candidates = self.reranker.rerank(query, candidates, top_k=rerank_limit)

        result = self.dedupe(candidates)
        logger.info(
            "retrieval complete; query_id=%s candidates=%d result=%d duration_ms=%.1f",
            qid,
            len(candidates),
            len(result),
            (time.perf_counter() - started) * 1000,
        )
        return result

    def dedupe(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunk_ids, first occurrence wins (drop the reranked duplicate)."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in candidates:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique.append(chunk)
        return unique