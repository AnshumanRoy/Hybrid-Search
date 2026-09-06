"""Application bootstrap (composition root).

Builds and wires the entire dependency graph. Framework imports are confined
to this module (and the equivalent api/cli entrypoints); services receive only
interfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.application.services import (
    ContextBuilder,
    IngestionService,
    LLMService,
    RAGService,
    RetrievalService,
)
from app.application.services.auth_service import AuthService
from app.application.services.document_service import DocumentService
from app.application.services.rate_limiting import RateLimiter
from app.application.services.rewrite import RewriteService
from app.common.cache import Cache, InMemoryCache
from app.common.cache.generation import (
    IndexGeneration,
    InMemoryIndexGeneration,
    RedisIndexGeneration,
)
from app.common.cache.redis import RedisCache
from app.common.cache.single_flight import RedisSingleFlight, ThreadingSingleFlight
from app.common.configuration.settings import Settings
from app.common.configuration.settings import CacheBackend, LLMProvider, VectorCacheBackend, VectorStoreBackend
from app.common.file_sources import LocalFileSource
from app.common.logging import setup_logging
from app.domain.exceptions import ConfigurationError
from app.domain.interfaces import Reranker
from app.infrastructure.chunkers.sentence import SentenceAwareChunker
from app.infrastructure.document_parsers.pdf import PDFParser
from app.infrastructure.document_parsers.router import DocumentParserRouter
from app.infrastructure.document_parsers.text import TextParser
from app.infrastructure.embedders.sentence_transformers import SentenceTransformerEmbedder
from app.infrastructure.llm.echo import EchoLLM
from app.infrastructure.llm.nemotron import NemotronNIMClient
from app.infrastructure.llm.ollama import OllamaLLM
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.persister import IndexPersister
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository
from app.infrastructure.persistence.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository
from app.infrastructure.rerankers.cross_encoder import CrossEncoderReranker
from app.infrastructure.retrievers.bm25 import BM25Retriever
from app.infrastructure.retrievers.hybrid import HybridRetriever
from app.infrastructure.rewriters.heuristic import HeuristicQueryRewriter
from app.infrastructure.rewriters.llm import LLMQueryRewriter
from app.infrastructure.vector_stores.caching import CachingVectorStore
from app.infrastructure.vector_stores.chroma import ChromaVectorStore
from app.infrastructure.vector_stores.pgvector import PgVectorStore

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    parser: object
    chunker: SentenceAwareChunker
    embedder: object
    database: Database
    repository: SQLAlchemyDocumentRepository
    users: SQLAlchemyUserRepository
    vector_store: object
    vector_cache: Cache | None
    vector_cache_generation: IndexGeneration | None
    lexical: BM25Retriever
    persister: IndexPersister
    hybrid_retriever: HybridRetriever
    reranker: object | None
    retrieval_service: RetrievalService
    context_builder: ContextBuilder
    llm: object
    llm_providers: dict[str, object]
    llm_service: LLMService
    rewrite_service: RewriteService
    rewrite_cache: Cache | None
    rag_service: RAGService
    ingestion_service: IngestionService
    auth_service: AuthService
    document_service: DocumentService
    rate_limiter: RateLimiter
    source: LocalFileSource


def build_container(
    settings: Settings | None = None,
    *,
    with_reranker: bool = True,
    embedder: object | None = None,
    vector_store: object | None = None,
    llm: object | None = None,
    llm_providers: dict[str, object] | None = None,
    vector_cache: Cache | None = None,
    vector_cache_generation: IndexGeneration | None = None,
) -> Container:
    settings = settings or Settings()
    setup_logging(settings.log_level, json=settings.log_json)

    # --- Infrastructure ---
    database = Database.from_settings(settings)
    repository = SQLAlchemyDocumentRepository(database.session_factory)
    users = SQLAlchemyUserRepository(database.session_factory)

    vector_store = vector_store or _build_vector_store(settings, database)

    cache_backend, generation = _build_vector_cache(
        settings,
        injected_cache=vector_cache,
        injected_generation=vector_cache_generation,
    )
    if cache_backend is not None and generation is not None:
        single_flight = (
            ThreadingSingleFlight(max_entries=settings.vector_cache_max_entries)
            if settings.vector_cache_backend != VectorCacheBackend.REDIS or vector_cache is not None
            else RedisSingleFlight(_new_redis_client(settings), lease_seconds=settings.vector_cache_stampede_wait_seconds)
        )
        vector_store = CachingVectorStore(
            store=vector_store,
            cache=cache_backend,
            generation=generation,
            single_flight=single_flight,
            ttl_seconds=settings.vector_cache_ttl_seconds,
            stampede_wait_seconds=settings.vector_cache_stampede_wait_seconds,
        )
        logger.info(
            "vector-search cache enabled; backend=%s ttl=%ds",
            settings.vector_cache_backend.value,
            settings.vector_cache_ttl_seconds,
        )

    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device.value,
        )
    if embedder.dimension != settings.vector_dimension:
        logger.warning(
            "embedder dimension=%d differs from configured vector_dimension=%d; using embedder dimension",
            embedder.dimension,
            settings.vector_dimension,
        )

    lexical = BM25Retriever()

    parser = DocumentParserRouter([PDFParser(settings), TextParser()])
    chunker = SentenceAwareChunker(settings.chunk_size, settings.chunk_overlap)

    persister = IndexPersister(repository, vector_store, lexical, generation=generation)

    # --- Retrieval chain ---
    hybrid = HybridRetriever.from_settings(vector_store, embedder, lexical, settings)
    reranker: Reranker | None = None
    if with_reranker and settings.reranker_enabled:
        try:
            reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                batch_size=settings.reranker_batch_size,
                device=_resolve_device(settings.embedding_device.value),
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully offline
            logger.warning("reranker unavailable (offline?) — degradation to no-rerank: %s", exc)

    retrieval_service = RetrievalService(
        hybrid_search=hybrid,
        reranker=reranker,
        default_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        max_query_length=settings.max_query_length,
        fusion=hybrid.fusion,
    )

    # --- LLM ---
    llm_registry, default_model = _build_llm_registry(settings, injected_llm=llm)
    if llm_providers is not None:
        llm_registry.update(llm_providers)
    llm = llm_registry[default_model] if default_model in llm_registry else _build_llm(settings)
    cache = _build_general_cache(settings, prefix="llm:")
    llm_service = LLMService(
        llm=llm,
        cache=cache,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
        providers=llm_registry,
        default_model=default_model,
    )

    # --- Query rewriting (multi-query expansion). Built unconditionally so the
    # /rewrite endpoint works even when /query rewriting is disabled; the gate
    # (whether RAGService sees it) is decided above when constructing
    # RAGService.
    heuristic_rewriter = HeuristicQueryRewriter()
    llm_rewriter = LLMQueryRewriter(
        llm_service,
        default_model=settings.rewrite_llm_model,
        timeout_seconds=settings.rewrite_timeout_seconds,
    )
    rewrite_cache = _build_general_cache(settings, prefix="rewrite:")
    rewrite_service = RewriteService(
        llm_rewriter,
        heuristic_rewriter,
        cache=rewrite_cache,
        default_strategy=settings.query_rewrite_strategy.value,
        default_max_variants=settings.rewrite_max_variants,
    )

    context_builder = ContextBuilder(settings.context_max_chars)
    rag_service = RAGService(
        retrieval_service,
        context_builder,
        llm_service,
        rewrite_service=rewrite_service if settings.query_rewrite_enabled else None,
    )

    # --- Ingestion ---
    source = LocalFileSource(settings.resolve_data_dir(), extensions=settings.allowed_extensions)
    ingestion_service = IngestionService(source, parser, chunker, embedder, persister, settings)

    # --- Auth & document access ---
    auth_service = AuthService(users, settings)
    document_service = DocumentService(ingestion_service, repository, users)

    # --- Cost guardrail (per-user rate limiting on /query + /rewrite) ---
    rate_limiter = RateLimiter(
        _build_general_cache(settings, prefix="rl:") if settings.rate_limit_enabled else None,
        windows=[
            (settings.rate_limit_window_seconds, settings.rate_limit_max_requests),
            (86_400, settings.rate_limit_daily_max_requests),
        ],
    )

    return Container(
        settings=settings,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        database=database,
        repository=repository,
        users=users,
        vector_store=vector_store,
        vector_cache=cache_backend,
        vector_cache_generation=generation,
        lexical=lexical,
        persister=persister,
        hybrid_retriever=hybrid,
        reranker=reranker,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        llm=llm,
        llm_providers=llm_registry,
        llm_service=llm_service,
        rewrite_service=rewrite_service,
        rewrite_cache=rewrite_cache,
        rag_service=rag_service,
        ingestion_service=ingestion_service,
        auth_service=auth_service,
        document_service=document_service,
        rate_limiter=rate_limiter,
        source=source,
    )


def _build_vector_store(settings: Settings, database: Database):
    if settings.vector_store_backend == VectorStoreBackend.CHROMA:
        return ChromaVectorStore.from_settings(settings)

    if settings.vector_store_backend == VectorStoreBackend.PGVECTOR:
        if settings.database_url.startswith("sqlite"):
            raise ConfigurationError(
                "VECTOR_STORE_BACKEND=pgvector requires a PostgreSQL DATABASE_URL "
                "(e.g. postgresql+psycopg://user:pass@host:5432/db). "
                "See docker-compose.yml for a ready-to-run pgvector service."
            )
        return PgVectorStore.from_settings(settings, database.engine)

    raise ValueError(f"unknown vector store backend: {settings.vector_store_backend}")


def _new_redis_client(settings: Settings):
    import redis

    return redis.Redis.from_url(settings.vector_cache_redis_url, socket_connect_timeout=2, decode_responses=True)


def _build_general_cache(settings: Settings, *, prefix: str) -> Cache | None:
    """LLM-response + rewrite caches, gated by ``CACHE_ENABLED``.

    ``MEMORY`` is a bounded in-process TTL dict; ``REDIS`` shares values across
    replicas. Both honor the same fail-open contract: an unreachable backend
    degrades to uncached rather than breaking requests.
    """
    if not settings.cache_enabled:
        return None
    if settings.cache_backend == CacheBackend.MEMORY:
        return InMemoryCache(
            max_entries=settings.cache_max_entries,
            default_ttl_seconds=settings.cache_ttl_seconds,
        )
    if settings.cache_backend == CacheBackend.REDIS:
        try:
            import redis

            client = redis.Redis.from_url(
                settings.cache_redis_url, socket_connect_timeout=2, decode_responses=True
            )
            return RedisCache(client, prefix=prefix, ttl_seconds=settings.cache_ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - caching must never break requests
            logger.warning("redis cache unavailable (%s); degrading to uncached", exc)
            return None
    raise ValueError(f"unknown cache backend: {settings.cache_backend}")


def _build_vector_cache(
    settings: Settings,
    *,
    injected_cache: Cache | None,
    injected_generation: IndexGeneration | None,
) -> tuple[Cache | None, IndexGeneration | None]:
    """Build the vector-search cache + generation counter, or ``(None, None)``.

    Backend is chosen purely from settings (``VECTOR_CACHE_BACKEND``) unless a
    cache double was injected. Any failure (e.g. Redis unavailable) degrades to
    uncached with a warning — caching must never break requests.
    """
    if injected_cache is not None:
        return injected_cache, injected_generation or InMemoryIndexGeneration()

    if settings.vector_cache_backend == VectorCacheBackend.NONE:
        return None, None

    if settings.vector_cache_backend == VectorCacheBackend.MEMORY:
        return (
            InMemoryCache(
                max_entries=settings.vector_cache_max_entries,
                default_ttl_seconds=settings.vector_cache_ttl_seconds,
            ),
            InMemoryIndexGeneration(),
        )

    if settings.vector_cache_backend == VectorCacheBackend.REDIS:
        try:
            client = _new_redis_client(settings)
            return (
                RedisCache(client, prefix="vc:", ttl_seconds=settings.vector_cache_ttl_seconds),
                RedisIndexGeneration(client, key="vector_cache:index_generation"),
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully when Redis is down
            logger.warning(
                "redis vector cache unavailable (%s); falling back to uncached search",
                exc,
            )
            return None, None

    raise ValueError(f"unknown vector cache backend: {settings.vector_cache_backend}")


def _resolve_device(preferred: str) -> str:
    from app.infrastructure.embedders.sentence_transformers import resolve_device

    return resolve_device(preferred)


def _build_llm(settings: Settings):
    """Build the default backend selected by ``LLM_PROVIDER``."""
    if settings.llm_provider == LLMProvider.ECHO:
        return EchoLLM(temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
    if settings.llm_provider == LLMProvider.OLLAMA:
        return OllamaLLM(
            base_url=settings.llm_base_url or "http://localhost:11434",
            model=settings.llm_model or "llama3.1",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if settings.llm_provider in (LLMProvider.NEMOTRON,):
        return _build_nemotron(settings)
    if settings.llm_provider == LLMProvider.OPENAI:
        raise ConfigurationError(
            "LLM_PROVIDER=openai is not implemented yet. Add an OpenAI implementation "
            "returning an LLM-compatible object and register it in _build_llm."
        )
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")


def _build_nemotron(settings: Settings) -> NemotronNIMClient:
    if not settings.nvidia_api_key:
        raise ConfigurationError(
            "NVIDIA_API_KEY is required when LLM_PROVIDER=nemotron or NEMOTRON_ENABLED=true. "
            "Get a key from the NVIDIA NIM catalog and set it in .env."
        )
    return NemotronNIMClient(
        model=settings.nemotron_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nemotron_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.nemotron_max_tokens,
        timeout_seconds=settings.nemotron_timeout_seconds,
        max_retries=settings.nemotron_max_retries,
        enable_thinking=settings.nemotron_enable_thinking,
    )


def _build_llm_registry(settings: Settings, *, injected_llm: object | None = None) -> tuple[dict[str, object], str]:
    """Build the ``name -> LLM`` registry + the default model name.

    ``echo`` and ``ollama`` are always registered (their constructors are pure
    config — zero network I/O); ``nemotron`` is registered when it is the
    default provider or ``NEMOTRON_ENABLED=true`` (requires a key). An injected
    ``llm`` overrides the default backend only, so callers can substitute a
    fake for one backend without forcing real clients for the others.
    """
    if settings.llm_provider == LLMProvider.OPENAI and injected_llm is None:
        # Preserved from _build_llm: selecting the unimplemented backend fails fast.
        _build_llm(settings)

    default_model = settings.llm_provider.value
    providers: dict[str, object] = {
        "echo": EchoLLM(temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens),
        "ollama": OllamaLLM(
            base_url=settings.llm_base_url or "http://localhost:11434",
            model=settings.llm_model or "llama3.1",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
    }
    if settings.llm_provider == LLMProvider.NEMOTRON or settings.nemotron_enabled:
        providers["nemotron"] = _build_nemotron(settings)
    if injected_llm is not None:
        providers[default_model] = injected_llm
    elif default_model not in providers:
        providers[default_model] = _build_llm(settings)
    return providers, default_model