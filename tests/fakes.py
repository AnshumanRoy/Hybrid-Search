"""Test doubles: deterministic, framework-free implementations of the domain
Protocols so unit/integration tests never touch models, networks, or disks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.domain.entities import Chunk, EmbeddedChunk, RetrievedChunk
from app.domain.exceptions import VectorStoreError
from app.domain.value_objects import VectorQuery


def _seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class FakeEmbedder:
    """Deterministic hash-based L2-normalized vectors (dim=8)."""

    def __init__(self, model_name: str = "fake-model", dimension: int = 8, version: int = 1) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self.version = version

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        state = _seed(text)
        vector: list[float] = []
        for i in range(self._dimension):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            vector.append((state % 2000) / 1000.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeVectorStore:
    """Pure-Python in-memory vector store honoring the VectorStore protocol."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self._items: dict[str, EmbeddedChunk] = {}

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        for ec in chunks:
            if len(ec.embedding) != self._dimension:
                raise VectorStoreError(f"dimension mismatch: {len(ec.embedding)} != {self._dimension}")
            self._items[ec.chunk.chunk_id] = ec

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._items.pop(cid, None)

    def search(self, query: VectorQuery) -> list[RetrievedChunk]:
        if not self._items:
            return []
        model = query.embedding_model
        mismatched = {ec.embedding_model for ec in self._items.values()}
        if mismatched and model not in mismatched:
            raise VectorStoreError(f"query model '{model}' does not match stored models {mismatched}")

        q = query.query_embedding
        scored: list[tuple[float, EmbeddedChunk]] = []
        for ec in self._items.values():
            sim = _cosine(q, ec.embedding)
            if _matches_filters(ec, query):
                scored.append((sim, ec))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=ec.chunk.chunk_id,
                document_id=ec.chunk.document_id,
                filename=ec.chunk.filename,
                source_uri=ec.chunk.source_uri,
                page_number=ec.chunk.page_number,
                text=ec.chunk.text,
                metadata=ec.chunk.metadata,
                semantic_score=round(sim, 6),
                retrieval_metadata={"semantic": True},
                query_id=query.query_id,
            )
            for sim, ec in scored[: query.top_k]
        ]

    def get_count(self) -> int:
        return len(self._items)

    def info(self) -> dict:
        return {"count": len(self._items), "name": "fake", "embeddings_model": "fake-model"}


def _matches_filters(ec: EmbeddedChunk, query: VectorQuery) -> bool:
    if query.filters.document_ids is not None and ec.chunk.document_id not in query.filters.document_ids:
        return False
    for key, value in query.filters.exact.items():
        if ec.chunk.metadata.get(key) != value:
            return False
    for key, value in query.filters.must_not.items():
        if ec.chunk.metadata.get(key) == value:
            return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class NoopReranker:
    """Identity reranker that preserves whatever order/score it receives."""

    def __init__(self, keep_scores: bool = True) -> None:
        self.keep_scores = keep_scores

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        ordered = list(candidates)
        if top_k is not None:
            ordered = ordered[:top_k]
        return [c.model_copy(deep=True) for c in ordered]


FakeLLM = None  # EchoLLM used directly where needed


def make_chunk(
    document_id: str = "doc_1",
    chunk_id: str | None = None,
    text: str = "sample text",
    page: int | None = 1,
    index: int = 0,
    metadata: dict | None = None,
    filename: str | None = None,
) -> Chunk:
    from app.common.utils.doc_utils import compute_chunk_id

    return Chunk(
        chunk_id=chunk_id or compute_chunk_id(document_id, page, index),
        document_id=document_id,
        filename=filename or f"{document_id}.txt",
        source_uri=f"file:///{document_id}.txt",
        page_number=page,
        text=text,
        chunk_index=index,
        metadata=metadata or {"document_type": "policy", "department": "finance"},
    )


def make_embedded_chunk(chunk: Chunk, embedder: FakeEmbedder) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedder.embed([chunk.text])[0],
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        embedding_version=embedder.version,
    )