"""Hybrid retriever — combines semantic + lexical retrieval, fused by
composable strategies (weighted sum or Reciprocal Rank Fusion).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.common.configuration.settings import Settings
from app.common.utils.fusion import rrf_fusion, weighted_fusion
from app.domain.entities import RetrievedChunk
from app.domain.interfaces import Embedder, LexicalRetriever, VectorStore
from app.domain.value_objects import ConfigurableFusion, SearchFilters, VectorQuery

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses a semantic (vector) and a lexical (BM25) retriever."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        lexical: LexicalRetriever,
        fusion: ConfigurableFusion,
        embedding_version: int = 0,
    ) -> None:
        if fusion.mode not in ("weighted", "rrf"):
            raise ValueError(f"unsupported fusion mode: {fusion.mode}")
        self.vector_store = vector_store
        self.embedder = embedder
        self.lexical = lexical
        self.fusion = fusion
        self.embedding_version = embedding_version

    @classmethod
    def from_settings(
        cls,
        vector_store: VectorStore,
        embedder: Embedder,
        lexical: LexicalRetriever,
        settings: Settings,
    ) -> "HybridRetriever":
        return cls(
            vector_store=vector_store,
            embedder=embedder,
            lexical=lexical,
            fusion=ConfigurableFusion(
                mode=settings.hybrid_fusion.value,
                weight_semantic=settings.fusion_weight_semantic,
                rrf_k=settings.rrf_k,
                rrf_semantic_weight=settings.rrf_weight_semantic,
                rrf_lexical_weight=settings.rrf_weight_lexical,
            ),
            embedding_version=settings.embedding_version,
        )

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
        semantic_top_k = semantic_top_k or top_k
        lexical_top_k = lexical_top_k or max(top_k * 2, semantic_top_k)

        semantic_results = self._semantic(query, semantic_top_k, filters, query_id)
        lexical_results = self.lexical.retrieve(query, lexical_top_k, filters=filters)

        for chunk in lexical_results:
            if chunk.query_id is None:
                chunk.query_id = query_id

        if not semantic_results:
            return lexical_results[:top_k]
        if not lexical_results:
            return semantic_results[:top_k]

        if self.fusion.mode == "weighted":
            fused = weighted_fusion(
                semantic_results,
                lexical_results,
                weight_semantic=self.fusion.weight_semantic,
            )
        else:
            fused = rrf_fusion(
                semantic_results,
                lexical_results,
                k=self.fusion.rrf_k,
                semantic_weight=self.fusion.rrf_semantic_weight,
                lexical_weight=self.fusion.rrf_lexical_weight,
            )
        return fused[:top_k]

    def _semantic(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None,
        query_id: str | None,
    ) -> list[RetrievedChunk]:
        embedding = self.embedder.embed([query])[0]
        return self.vector_store.search(
            VectorQuery(
                query_id=query_id or "",
                query=query,
                query_embedding=embedding,
                embedding_model=self.embedder.model_name,
                embedding_version=self.embedding_version,
                top_k=top_k,
                filters=filters or SearchFilters(),
            )
        )