"""Integration: PDF → parser → chunker → embed → vector store → BM25 →
hybrid → rerank → prompt → EchoLLM → RAGResponse with citations.

No network calls: the embedder and reranker are deterministic fakes; EchoLLM
stands in for a real LLM; ChromaDB uses a temporary persistent directory.
"""

from __future__ import annotations

import pytest

from app.application.services.context import ContextBuilder
from app.application.services.llm_service import LLMService
from app.application.services.prompt import build_system_prompt, build_user_prompt, citations_from_chunks
from app.application.services.rag import RAGService
from app.application.services.retrieval import RetrievalService
from app.common.configuration.settings import HybridFusion, Settings
from app.domain.entities import FileReference
from app.domain.value_objects import ConfigurableFusion
from app.infrastructure.chunkers.sentence import SentenceAwareChunker
from app.infrastructure.document_parsers.pdf import PDFParser
from app.infrastructure.llm.echo import EchoLLM
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.persister import IndexPersister
from app.infrastructure.persistence.repositories.sqlalchemy_repository import SQLAlchemyDocumentRepository
from app.infrastructure.retrievers.bm25 import BM25Retriever
from app.infrastructure.retrievers.hybrid import HybridRetriever
from app.infrastructure.vector_stores.chroma import ChromaVectorStore
from tests.fakes import FakeEmbedder, NoopReranker


def _make_pdf_bytes(texts: list[str]) -> bytes:
    import fitz

    document = fitz.open()
    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def unused_engine():
    return None


def test_end_to_end_pipeline(tmp_path) -> None:
    settings = Settings(
        embedding_model="fake-model",
        vector_dimension=8,
        embedding_batch_size=2,
        vector_store_dir=tmp_path / "vs",
        database_url="sqlite://",
        data_dir=tmp_path / "data",
        chunk_size=100,
        chunk_overlap=20,
        cache_enabled=False,
        hybrid_fusion=HybridFusion.RRF,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build real components ----
    embedder = FakeEmbedder()
    database = Database.from_settings(settings)
    repository = SQLAlchemyDocumentRepository(database.session_factory)
    vector_store = ChromaVectorStore(tmp_path / "vs", settings.vector_collection_name, 8, embedding_model="fake-model")
    lexical = BM25Retriever()
    persister = IndexPersister(repository, vector_store, lexical)

    parser = PDFParser(settings)
    chunker = SentenceAwareChunker(settings.chunk_size, settings.chunk_overlap)

    # ---- 1. Ingest a real PDF ----
    pdf_bytes = _make_pdf_bytes(
        [
            "Acme refund policy states customers may return items within 30 days for a full refund.",
            "Acme employees accrue 15 days of paid annual leave each calendar year.",
        ]
    )
    ref = FileReference(path="policy.pdf", filename="policy.pdf", uri="file:///policy.pdf", size_bytes=len(pdf_bytes))
    doc = parser.parse(ref, pdf_bytes)
    assert doc.page_count == 2

    chunks = chunker.chunk(doc)
    assert chunks

    embedded = [
        _embedded(c, embedder) for c in chunks
    ]
    persister.persist_document(doc, embedded)
    assert vector_store.get_count() == len(embedded)
    assert lexical.count() == len(embedded)
    assert repository.get_document(doc.document_id) is not None

    # ---- 2. Hybrid retrieval ----
    hybrid = HybridRetriever(
        vector_store, embedder, lexical, ConfigurableFusion(mode="rrf", rrf_k=60), embedding_version=1
    )
    retrieval_service = RetrievalService(hybrid, reranker=NoopReranker(), default_top_k=3)
    hits = retrieval_service.retrieve("What is the refund policy?")
    assert hits
    assert hits[0].document_id == doc.document_id

    # ---- 3. Context + prompt ----
    context = ContextBuilder().build(hits)
    assert "Refund" in context or "refund" in context
    prompt = build_user_prompt(context, "What is the refund policy?")
    assert build_system_prompt().lower().count("untrusted") >= 1

    # ---- 4. Generation (echo) + citations ----
    rag = RAGService(retrieval_service, ContextBuilder(), LLMService(EchoLLM()))
    response = rag.answer("What is the refund policy?", top_k=3)
    assert response.answer
    assert citations_from_chunks(hits)


def _embedded(chunk, embedder):
    from app.domain.entities import EmbeddedChunk

    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedder.embed([chunk.text])[0],
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        embedding_version=1,
    )


def test_embedding_model_mismatch_raises(tmp_path) -> None:
    """Search against a store built with a different model must fail loudly."""
    settings = Settings(
        embedding_model="fake-model",
        vector_dimension=8,
        vector_store_dir=tmp_path / "vs2",
        database_url="sqlite://",
        data_dir=tmp_path / "data2",
        cache_enabled=False,
    )
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "some text")
    data = pdf.tobytes()
    pdf.close()

    ref = FileReference(path="x.pdf", filename="x.pdf")
    doc = PDFParser(settings).parse(ref, data)
    chunks = SentenceAwareChunker(100, 10).chunk(doc)

    embedder_a = FakeEmbedder(model_name="model-a")
    store = ChromaVectorStore(tmp_path / "vs2", settings.vector_collection_name, 8, embedding_model="model-a")
    store.upsert([_embedded(c, embedder_a) for c in chunks])

    embedder_b = FakeEmbedder(model_name="model-b")
    from app.domain.exceptions import VectorStoreError
    from app.domain.value_objects import VectorQuery

    with pytest.raises(VectorStoreError):
        store.search(
            VectorQuery(
                query_id="q1",
                query="test",
                query_embedding=embedder_b.embed(["test"])[0],
                embedding_model="model-b",
                embedding_version=1,
                top_k=3,
            )
        )