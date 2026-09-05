# Architecture

The system is a layered pipeline divided into four concentric layers selected by dependency
inversion: **domain**, **application**, **infrastructure**, and **delivery** (API/CLI/composition
root). Services hold references only to interfaces; all concrete wiring happens in one module,
`app/bootstrap.py`.

```
┌─────────────────────────────── delivery ───────────────────────────────┐
│  api/routes/* (FastAPI)      __main__ (CLI)       bootstrap (wiring)   │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │ depends on
┌─────────────────────────────── application ────────────────────────────┐
│  RAGService  RetrievalService  IngestionService  LLMService            │
│  ContextBuilder  PromptBuilder  (orchestration, no framework imports)  │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │ depends on (all Protocols)
┌─────────────────────────────── domain ─────────────────────────────────┐
│  entities / value_objects / exceptions                                  │
│  interfaces.py     │            │              │                        │
└────────────────────┼────────────┼──────────────┼────────────────────────┘
                     ▼            ▼              ▼
┌─────────────────────────────── infrastructure ─────────────────────────┐
│  DocumentParserRouter  Chunker  Embedder  VectorStore  BM25  Reranker   │
│  SQLAlchemyRepository  IndexPersister  ChromaVectorStore  LLM backends  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Layering rules

- **Domain** (`app/domain/`) — the vocabulary: `ParsedDocument`, `Chunk`, `EmbeddedChunk`,
  `RetrievedChunk`, `Citation`, `RAGResponse`, `SearchFilters`, `ConfigurableFusion`; exceptions
  hierarchy rooted at `RAGError`; and the `Protocol` interfaces every collaborator implements:
  `DocumentParser`, `Chunker`, `Embedder`, `VectorStore`, `DocumentRepository`, `LexicalRetriever`,
  `Reranker`, `LLM`, `FileSource`.
- **Application** (`app/application/`) — pure orchestration. Services receive interfaces and
  combination-config and never import a framework. This is where test doubles are also injected.
- **Infrastructure** (`app/infrastructure/`) — concrete adapters: PyMuPDF/Text parsers,
  sentence-aware chunker, sentence-transformers embedder, Chroma vector store,
  BM25/hybrid retrievers, cross-encoder reranker, SQLAlchemy persistence, echo/ollama LLMs.
- **Delivery** — `api/` exposes REST endpoints; `__main__.py` exposes the CLI; `bootstrap.py`
  is the single composition root.

## Dependency graph (runtime)

```mermaid
graph LR
  subgraph Delivery
    API
    CLI
    Bootstrap
  end

  subgraph Application
    IS[IngestionService]
    RS[RetrievalService]
    RAG[RAGService]
    LS[LLMService]
    CB[ContextBuilder]
  end

  subgraph Infrastructure
    Parser[DocumentParserRouter]
    Chunk[Chunker]
    Emb[Embedder]
    VS[ChromaVectorStore]
    BM[BM25Retriever]
    H[HybridRetriever]
    RR[Reranker]
    DB[(SQLite)]
    IP[IndexPersister]
    LLM[LLM backend]
  end

  Bootstrap --> Parser
  Bootstrap --> Chunk
  Bootstrap --> Emb
  Bootstrap --> VS
  Bootstrap --> BM
  Bootstrap --> DB
  Bootstrap --> RR
  Bootstrap --> LLM

  IS --> Parser
  IS --> Chunk
  IS --> Emb
  IS --> IP

  IP --> DB
  IP --> VS
  IP --> BM

  H --> VS
  H --> Emb
  H --> BM

  RS --> H
  RS --> RR
  RAG --> RS
  RAG --> CB
  RAG --> LS
  LS --> LLM

  API --> RAG
  API --> IS
  CLI --> RAG
  CLI --> IS
  CLI --> EP[EvaluationRunner]
```

## Request/response flow

```mermaid
sequenceDiagram
    participant C as Client/CLI
    participant RAG as RAGService
    participant RS as RetrievalService
    participant H as HybridRetriever
    participant RR as Reranker
    participant CB as ContextBuilder
    participant LS as LLMService
    participant L as LLM

    C->>RAG: answer(query, top_k)
    RAG->>RS: retrieve(query)
    RS->>H: search(query)
    H->>H: embed query + BM25 lexical search
    H->>RS: fused + ranked RetrievedChunk[]
    RS->>RR: rerank(top candidates)
    RR->>RS: reranked top_k
    RS->>RAG: RetrievedChunk[]
    RAG->>CB: build(chunks, max_chars)
    CB->>RAG: context blocks (with citations)
    RAG->>LS: complete(prompt, cache_key)
    LS->>L: generate
    L->>LS: response
    LS->>RAG: answer text
    RAG->>C: RAGResponse(answer, citations, metadata)
```

## Ingestion flow

```mermaid
sequenceDiagram
    participant S as FileSource
    participant IS as IngestionService
    participant P as ParserRouter
    participant CH as Chunker
    participant E as Embedder
    participant RP as IndexPersister
    participant DB as SQLite
    participant VS as Chroma
    participant BM as BM25

    S->>IS: files (or bytes via API)
    IS->>IS: content_hash() per file
    IS->>RP: find_by_hash(hash)
    alt unchanged
        RP-->>IS: exists → [skipped]
    else new/changed
        IS->>P: parse(file) → ParsedDocument
        P-->>IS: pages + text
        IS->>CH: chunk(pages) → Chunk[]
        CH-->>IS: chunks
        IS->>E: embed(texts) → vectors
        E-->>IS: EmbeddedChunk[]
        IS->>RP: replace_document(...)
        RP->>DB: upsert document + chunks
        RP->>VS: add/delete vectors
        RP->>BM: add/delete terms
        IS-->>S: [indexed]
    end
```

## Composability & testability

`build_container(settings=None, *, with_reranker=True, embedder=None, vector_store=None,
llm=None)` accepts injected doubles for the three heavy pieces (embedder, vector store, LLM).
Integration tests inject `FakeEmbedder`/real Chroma; the `echo` LLM keeps the whole pipeline
offline-testable. Every service can also be constructed directly with fakes for unit tests —
there are no hidden globals or module-level singletons inside components (only
`get_settings()` exists at the composition root).