"""Retrieval orchestration service: query → hybrid search → rerank → top N."""

from __future__ import annotations

import logging
import time
import uuid

from app.common.utils.fusion import rrf_merge_lists, weighted_merge_lists
from app.domain.entities import RetrievedChunk
from app.domain.exceptions import RetrievalError
from app.domain.interfaces import HybridSearch, Reranker
from app.domain.value_objects import ConfigurableFusion, RewrittenQuery, SearchFilters

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: Reranker | None = None,
        default_top_k: int = 10,
        rerank_top_k: int | None = None,
        max_query_length: int = 1000,
        fusion: ConfigurableFusion | None = None,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.rerank_top_k = rerank_top_k
        self.max_query_length = max_query_length
        self.fusion = fusion

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

    def retrieve_multi(
        self,
        rewritten: RewrittenQuery,
        top_k: int | None = None,
        *,
        filters: SearchFilters | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Multi-query retrieval: fan out over variants, fuse, rerank, dedupe.

        Sequential fan-out (deterministic; avoids assuming thread-safety of the
        Chroma client and SentenceTransformer). The **same** ``filters`` reach
        every variant — access control must never vary by variant — and reuses
        the configured ``ConfigurableFusion`` one level up.
        """
        top_k = top_k or self.default_top_k
        if self.reranker is not None and top_k == 1:
            top_k = self.rerank_top_k or top_k

        original = (rewritten.original or "").strip()
        if not original:
            raise RetrievalError("query must not be empty")
        if len(original) > self.max_query_length:
            raise RetrievalError(f"query exceeds maximum length of {self.max_query_length} characters")
        if not rewritten.variants:
            raise RetrievalError("rewritten query has no variants")

        qid = query_id or uuid.uuid4().hex
        started = time.perf_counter()

        per_variant: list[list[RetrievedChunk]] = []
        for variant in rewritten.variants:
            candidates = self.hybrid_search.retrieve(variant, top_k, filters=filters, query_id=qid)
            if candidates:
                per_variant.append(candidates)
        if not per_variant:
            logger.info("multi-query retrieval returned no candidates; query_id=%s", qid)
            return []

        merged = self._merge_variants(per_variant)

        if self.reranker is not None:
            rerank_limit = self.rerank_top_k or max(top_k, len(merged))
            merged = [c for c in merged if c.text.strip()]
            if merged:
                merged = self.reranker.rerank(rewritten.original, merged, top_k=rerank_limit)

        result = self.dedupe(merged)
        logger.info(
            "multi-query retrieval complete; query_id=%s variants=%d candidates=%d result=%d duration_ms=%.1f",
            qid,
            len(rewritten.variants),
            len(merged),
            len(result),
            (time.perf_counter() - started) * 1000,
        )
        return result

    def _merge_variants(self, lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        """Cross-variant merge reusing the configured fusion primitive."""
        mode = self.fusion.mode if self.fusion is not None else "rrf"
        if mode == "weighted":
            return weighted_merge_lists(lists)
        k = self.fusion.rrf_k if self.fusion is not None else 60
        return rrf_merge_lists(lists, k=k)

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