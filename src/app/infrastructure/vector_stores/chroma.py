"""ChromaDB vector store (persistent).

Chroma-specific objects never escape this module; the rest of the app works
against the ``VectorStore`` protocol.

Embedding versioning: the store is *partitioned* by embedding model name.
Search declares the model/version it expects and the store refuses to search
a collection whose recorded model differs — this prevents silently mixing
embeddings produced by different models.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.common.configuration.settings import Settings
from app.domain.entities import EmbeddedChunk, RetrievedChunk
from app.domain.exceptions import RetrievalError, VectorStoreError
from app.domain.value_objects import SearchFilters, VectorQuery

logger = logging.getLogger(__name__)

_METADATA_MODEL_KEY = "embedding_model"
_METADATA_VERSION_KEY = "embedding_version"
_METADATA_DIM_KEY = "embedding_dimension"
_METADATA_CHUNK_KEY = "chunk_id"
_METADATA_DOCUMENT_KEY = "document_id"
_METADATA_PAGE_KEY = "page_number"
_METADATA_FILENAME_KEY = "filename"
_METADATA_SOURCE_URI_KEY = "source_uri"

# Sentinel for an empty allow-list: Chroma rejects empty "$in" values, so a
# filter that must match nothing is expressed as equality against an
# impossible document id.
_IMPOSSIBLE_DOCUMENT_ID = "__no_document_matches__"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


class ChromaVectorStore:
    """ChromaDB-backed persistent vector store satisfying the ``VectorStore`` protocol."""

    def __init__(
        self,
        path: Path,
        collection_name: str,
        dimension: int,
        embedding_model: str,
        distance: str = "cosine",
    ) -> None:
        self._dimension = dimension
        self._embedding_model = embedding_model
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("chromadb is not installed") from exc

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=_slugify(collection_name),
            metadata={"hnsw:space": distance},
        )
        stored_meta = self._collection.metadata or {}
        persisted_model = stored_meta.get("embedding_model")
        if persisted_model not in (None, self._embedding_model):
            persisted_dim = stored_meta.get("embedding_dimension")
            raise VectorStoreError(
                f"collection '{collection_name}' already exists for embedding model "
                f"'{persisted_model}' (dim={persisted_dim}); refusing to mix incompatible vectors"
            )
        # Chroma forbids changing hnsw:space after creation, so strip config
        # keys before merging our own metadata.
        persist_meta = {k: v for k, v in stored_meta.items() if k != "hnsw:space"}
        self._collection.modify(metadata={**persist_meta, "embedding_model": self._embedding_model})

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChromaVectorStore":
        return cls(
            path=settings.vector_store_dir,
            collection_name=settings.vector_collection_name,
            dimension=settings.vector_dimension,
            embedding_model=settings.embedding_model,
            distance=settings.vector_distance,
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        model = chunks[0].embedding_model
        version = chunks[0].embedding_version
        if model != self._embedding_model:
            raise VectorStoreError(
                f"refusing to upsert chunks embedded with '{model}' into a store configured for "
                f"'{self._embedding_model}'"
            )
        stored_meta = self._collection.metadata or {}
        if stored_meta.get("embedding_model") not in (None, model):
            raise VectorStoreError("refusing to upsert embeddings from a different model into the collection")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for ec in chunks:
            if len(ec.embedding) != self._dimension:
                raise VectorStoreError(
                    f"embedding dimension mismatch: chunk '{ec.chunk.chunk_id}' has "
                    f"{len(ec.embedding)} dims, expected {self._dimension}"
                )
            ids.append(ec.chunk.chunk_id)
            embeddings.append(ec.embedding)
            documents.append(ec.chunk.text)
            metadatas.append(
                {
                    _METADATA_MODEL_KEY: ec.embedding_model,
                    _METADATA_VERSION_KEY: ec.embedding_version,
                    _METADATA_DIM_KEY: len(ec.embedding),
                    _METADATA_CHUNK_KEY: ec.chunk.chunk_id,
                    _METADATA_DOCUMENT_KEY: ec.chunk.document_id,
                    _METADATA_PAGE_KEY: ec.chunk.page_number if ec.chunk.page_number is not None else -1,
                    _METADATA_FILENAME_KEY: ec.chunk.filename,
                    _METADATA_SOURCE_URI_KEY: ec.chunk.source_uri or "",
                    **{f"md_{k}": v for k, v in ec.chunk.metadata.items()},
                }
            )

        try:
            self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        except Exception as exc:
            raise VectorStoreError(f"chroma upsert failed for {len(chunks)} chunks: {exc}") from exc

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        try:
            self._collection.delete(ids=list(chunk_ids))
        except Exception as exc:
            raise VectorStoreError(f"chroma delete failed: {exc}") from exc

    def search(self, query: VectorQuery) -> list[RetrievedChunk]:
        if query.embedding_model != self._embedding_model:
            raise VectorStoreError(
                f"query embedding model '{query.embedding_model}' does not match store model "
                f"'{self._embedding_model}'; re-index with the matching embedder"
            )
        if len(query.query_embedding) != self._dimension:
            raise RetrievalError(
                f"query embedding has {len(query.query_embedding)} dims but store expects {self._dimension}"
            )

        where = self._build_filter(query.filters)
        try:
            result = self._collection.query(query_embeddings=[query.query_embedding], n_results=query.top_k, where=where)
        except Exception as exc:
            raise RetrievalError(f"vector search failed: {exc}") from exc

        retrieved: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else []
        metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
        documents = result.get("documents", [[]])[0] if result.get("documents") else []

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            distance = distances[i] if i < len(distances) else None
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get(_METADATA_DOCUMENT_KEY, "")),
                    filename=str(meta.get(_METADATA_FILENAME_KEY, "")),
                    source_uri=meta.get(_METADATA_SOURCE_URI_KEY) or None,
                    page_number=meta.get(_METADATA_PAGE_KEY) if meta.get(_METADATA_PAGE_KEY) not in (None, -1) else None,
                    text=(documents[i] if i < len(documents) and documents[i] else ""),
                    metadata={
                        k.removeprefix("md_"): v
                        for k, v in meta.items()
                        if k.startswith("md_") and k not in (_METADATA_MODEL_KEY,)
                    },
                    semantic_score=self._to_similarity(distance),
                    retrieval_metadata={"semantic": True, "distance": distance},
                    query_id=query.query_id,
                )
            )
        return retrieved

    def get_count(self) -> int:
        return self._collection.count()

    def info(self) -> dict[str, Any]:
        return {
            "count": self._collection.count(),
            "name": self._collection.name,
            "embeddings_model": (self._collection.metadata or {}).get("embedding_model"),
            "hnsw_space": (self._collection.metadata or {}).get("hnsw:space"),
        }

    def _build_filter(self, filters: SearchFilters) -> dict | None:
        if filters.is_empty():
            return None
        clauses: list[dict] = []

        if filters.document_ids is not None:
            if filters.document_ids:
                clauses.append({_METADATA_DOCUMENT_KEY: {"$in": list(filters.document_ids)}})
            else:
                clauses.append({_METADATA_DOCUMENT_KEY: {"$eq": _IMPOSSIBLE_DOCUMENT_ID}})
        for key, value in filters.exact.items():
            clauses.append({f"md_{key}": {"$eq": value}})
        for key, value in filters.must_not.items():
            clauses.append({f"md_{key}": {"$ne": value}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _to_similarity(distance: float | None) -> float | None:
        if distance is None:
            return None
        return max(0.0, min(1.0, 1.0 - distance))