"""Value objects shared across domain and application layers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Transport-agnostic metadata filters for retrieval.

    Semantics: ``document_ids`` restricts results to chunks belonging to the
    listed document ids (pushed down into every store: vector + lexical);
    all conditions under ``exact`` must match the chunk metadata exactly;
    ``must_not`` matches are excluded.  ``limit_per_document`` optionally caps
    how many chunks are returned per document.
    """

    document_ids: list[str] | None = None
    exact: dict[str, Any] = Field(default_factory=dict)
    must_not: dict[str, Any] = Field(default_factory=dict)
    limit_per_document: int | None = None

    def is_empty(self) -> bool:
        return (
            self.document_ids is None
            and not self.exact
            and not self.must_not
            and self.limit_per_document is None
        )


class VectorQuery(BaseModel):
    """Plays the role of a typed query for the vector store."""

    query_id: str
    query: str
    query_embedding: list[float]
    embedding_model: str
    embedding_version: int
    top_k: int
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ConfigurableFusion(BaseModel):
    """Fusion configuration passed to the hybrid retriever."""

    mode: str  # "weighted" | "rrf"
    weight_semantic: float = 0.5
    rrf_k: int = 60
    rrf_semantic_weight: float = 1.0
    rrf_lexical_weight: float = 1.0


__all__ = ["SearchFilters", "VectorQuery", "ConfigurableFusion"]