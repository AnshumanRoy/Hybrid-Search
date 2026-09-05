"""Application settings.

Typed configuration loaded from environment variables / ``.env`` files.
Kept framework/locator agnostic: no global singletons, just a factory that
produces an immutable ``Settings`` object at the composition root.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_project_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``pyproject.toml``."""
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
        if parent == parent.parent:
            break
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root(Path(__file__))


class LLMProvider(str, Enum):
    """Supported LLM providers (implemented backends).

    Add providers to this enum *and* provide a matching implementation behind
    the ``LLM`` protocol to extend the system without touching orchestration.
    """

    ECHO = "echo"
    OLLAMA = "ollama"
    OPENAI = "openai"  # interface-only; registers a clear ConfigurationError until key configured


class EmbeddingDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class HybridFusion(str, Enum):
    WEIGHTED = "weighted"
    RRF = "rrf"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class Settings(BaseSettings):
    """Typed, immutable application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    embedding_device: EmbeddingDevice = EmbeddingDevice.AUTO
    embedding_version: int = Field(default=1, ge=0)

    # Vector store
    vector_store_backend: VectorStoreBackend = VectorStoreBackend.CHROMA
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    vector_dimension: int = Field(default=384, ge=1)
    vector_distance: str = "cosine"

    # Various collection/dimensions names derived from the model to avoid
    # silently mixing vectors from different embedding models.
    @property
    def vector_collection_name(self) -> str:
        model_slug = self.embedding_model.replace("/", "_").replace(".", "_")
        return f"hybrid_search_{model_slug}"

    @property
    def vector_table_name(self) -> str:
        model_slug = self.embedding_model.replace("/", "_").replace(".", "_")
        return f"hybrid_search_{model_slug}"

    # Database
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'rag.db'}"

    # Document sources
    data_dir: Path = PROJECT_ROOT / "data"
    allowed_extensions: list[str] = Field(default_factory=lambda: ["pdf", "txt", "md"])
    max_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_documents_per_ingest_batch: int = Field(default=100, ge=1)

    # Chunking
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    semantic_top_k: int = Field(default=10, ge=1)
    bm25_top_k: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=5, ge=0)
    hybrid_fusion: HybridFusion = HybridFusion.RRF
    fusion_weight_semantic: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rrf_weight_semantic: float = Field(default=1.0, ge=0.0)
    rrf_weight_lexical: float = Field(default=1.0, ge=0.0)

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_enabled: bool = True

    # LLM
    llm_provider: LLMProvider = LLMProvider.ECHO
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0)

    # Context construction
    context_max_chars: int = Field(default=8000, ge=100)

    # Caching
    cache_enabled: bool = True
    cache_max_entries: int = Field(default=4096, ge=1)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_query_length: int = Field(default=1000, ge=1)
    max_pdf_pages: int = Field(default=2000, ge=1)
    max_pdf_page_count: int = Field(default=2000, ge=1)

    # Auth
    auth_secret_key: str = "dev-only-change-me"
    auth_token_ttl_seconds: int = Field(default=86_400, ge=1)

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Evaluation
    eval_top_ks: list[int] = Field(default_factory=lambda: [1, 5, 10])

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in logging._nameToLevel:
            raise ValueError(f"Invalid LOG_LEVEL: {v}")
        return v

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, v: list[str]) -> list[str]:
        return [e.lower().lstrip(".") for e in v]

    @field_validator("vector_dimension")
    @classmethod
    def _validate_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vector_dimension must be positive")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def resolve_data_dir(self) -> Path:
        return self.data_dir.resolve()

    def sqlalchemy_url(self) -> str:
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-level settings (cached, single construction).

    The cache is a convenience at the composition root only; components never
    call ``get_settings()`` themselves.
    """
    return Settings()


def load_settings(settings: Settings | None = None, **overrides) -> Settings:
    """Build a Settings instance for tests / alternative composition roots."""
    if settings is not None:
        return settings
    return Settings(**overrides)