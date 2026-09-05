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
from app.common.cache import InMemoryCache
from app.common.configuration.settings import Settings
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
from app.infrastructure.llm.ollama import OllamaLLM
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.persister import IndexPersister
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository
from app.infrastructure.rerankers.cross_encoder import CrossEncoderReranker
from app.infrastructure.retrievers.bm25 import BM25Retriever
from app.infrastructure.retrievers.hybrid import HybridRetriever
from app.infrastructure.vector_stores.chroma import ChromaVectorStore
from app.common.configuration.settings import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    parser: object
    chunker: SentenceAwareChunker
    embedder: object
    database: Database
    repository: SQLAlchemyDocumentRepository
    vector_store: object
    lexical: BM25Retriever
    persister: IndexPersister
    hybrid_retriever: HybridRetriever
    reranker: object | None
    retrieval_service: RetrievalService
    context_builder: ContextBuilder
    llm: object
    llm_service: LLMService
    rag_service: RAGService
    ingestion_service: IngestionService
    source: LocalFileSource


def build_container(
    settings: Settings | None = None,
    *,
    with_reranker: bool = True,
    embedder: object | None = None,
    vector_store: object | None = None,
    llm: object | None = None,
) -> Container:
    settings = settings or Settings()
    setup_logging(settings.log_level, json=settings.log_json)

    # --- Infrastructure ---
    database = Database.from_settings(settings)
    repository = SQLAlchemyDocumentRepository(database.session_factory)

    vector_store = vector_store or ChromaVectorStore.from_settings(settings)

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

    persister = IndexPersister(repository, vector_store, lexical)

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
    )

    # --- LLM ---
    llm = llm or _build_llm(settings)
    cache = InMemoryCache(max_entries=settings.cache_max_entries) if settings.cache_enabled else None
    llm_service = LLMService(
        llm=llm,
        cache=cache,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
    )

    context_builder = ContextBuilder(settings.context_max_chars)
    rag_service = RAGService(retrieval_service, context_builder, llm_service)

    # --- Ingestion ---
    source = LocalFileSource(settings.resolve_data_dir(), extensions=settings.allowed_extensions)
    ingestion_service = IngestionService(source, parser, chunker, embedder, persister, settings)

    return Container(
        settings=settings,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        database=database,
        repository=repository,
        vector_store=vector_store,
        lexical=lexical,
        persister=persister,
        hybrid_retriever=hybrid,
        reranker=reranker,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        llm=llm,
        llm_service=llm_service,
        rag_service=rag_service,
        ingestion_service=ingestion_service,
        source=source,
    )


def _resolve_device(preferred: str) -> str:
    from app.infrastructure.embedders.sentence_transformers import resolve_device

    return resolve_device(preferred)


def _build_llm(settings: Settings):
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
    if settings.llm_provider == LLMProvider.OPENAI:
        raise ConfigurationError(
            "LLM_PROVIDER=openai is not implemented yet. Add an OpenAI implementation "
            "returning an LLM-compatible object and register it in _build_llm."
        )
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")