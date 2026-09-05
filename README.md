# Hybrid Search — Production RAG Pipeline

A modular, production-oriented **retrieval-augmented generation (RAG)** pipeline built in Python.
It ingests PDFs/plain-text files, splits them into sentence-aware chunks, and serves groundable
answers backed by **hybrid retrieval** (dense embeddings + BM25 lexical) with optional
**cross-encoder reranking**, structured **citations**, an evaluation harness, a FastAPI server,
a CLI, and Docker packaging.

```
documents → parse → chunk → embed → SQLite + Chroma + BM25
                                      ↓
 query ─→ hybrid retrieval ─→ rerank ─→ context/prompt ─→ LLM ─→ answer + citations
```

---

## Features

- **Layered, dependency-inverted architecture** — services depend on `Protocol` interfaces
  (`app/domain/interfaces.py`); only the composition root (`app/bootstrap.py`) touches frameworks.
- **Multi-format ingestion** — PDF (PyMuPDF) and plain text (`.txt`/`.md`), routed by extension,
  with strict size/extension guards and watch-for-traversal file sources.
- **Sentence-aware chunking** — configurable window and overlap so sentence context is never
  sliced mid-sentence.
- **Three-way index** — SQLite (SQLAlchemy) for document/chunk metadata, Chroma for dense vectors,
  BM25 for lexical scoring. Ingest is **idempotent and incremental** via content hashes: unchanged
  files are skipped, changed files are re-indexed atomically.
- **Hybrid retrieval** — merge semantic + BM25 scores with configurable **RRF** (default) or
  **weighted** fusion; optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker.
- **Grounded generation** — context assembly with explicit citations (filename, page, chunk),
  deterministic prompt templates that treat retrieved context as *untrusted*.
- **Evaluation harness** — retrieval metrics (recall/precision@K, MRR, nDCG) plus generation
  metrics (faithfulness, answer/context relevance). 96 tests included (unit + integration).
- **API + CLI** — FastAPI (`/health`, `/query`, `/documents/ingest*`, `/documents`) and a CLI for
  `ingest`, `query`, and `evaluate`.
- **Fail-soft design** — missing/offline reranker, LLM, or embedding model degrades gracefully
  (never crashes the pipeline), with an `echo` LLM for offline smoke tests.

---

## Quickstart

> Python 3.10+ required. On Windows use the provided venv: `env\Scripts\python.exe`.

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure (optional). Copy .env.example -> .env and adjust.
cp .env.example .env

# 3. Ingest the bundled sample documents (acme_handbook.pdf + .txt)
python -m app ingest

# 4. Ask a question (default LLM_PROVIDER=echo prints a deterministic placeholder)
python -m app query "What is Acme's refund window for physical products?"

# 5. Serve the API
uvicorn app.app:app --reload
curl http://127.0.0.1:8000/health
```

### Offline / first-run note

Embedding and reranker models are downloaded from the Hugging Face Hub on first use.
To force offline operation after models are cached (useful for CI/tests):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

To skip the (optional) reranker entirely:

```bash
export RERANKER_ENABLED=false
```

---

## CLI

| Command | Description |
| --- | --- |
| `python -m app ingest [path]` | Ingest every allowed file in `path`; defaults to `data/sample`. Prints per-file status (`indexed` / `skipped` / `failed`). |
| `python -m app query "question" [--top-k N]` | Run the full RAG pipeline and print the answer with citations and timing metadata. |
| `python -m app evaluate [dataset] [--generate] [--top-k N]` | Run retrieval (and optional generation) evaluation metrics against a dataset. |

## API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/health` | — | `{status, model, document_count}` |
| POST | `/query` | `{"query", "top_k"}` | `{answer, citations[], metadata{query_id, retrieval_ms, generation_ms}}` |
| POST | `/documents/ingest` | multipart `file` | `list[{document_id, filename, status, chunks_embedded, error, duration_ms}]` |
| POST | `/documents/ingest-dir` | `{"path"}` | same as above (server-side directory ingest) |
| GET | `/documents` | — | `list[{document_id, filename, status, pages, chunks, ingested_at, content_hash}]` |

Interactive docs at `/docs` (Swagger UI).

---

## Configuration

Settings are typed via pydantic-settings and read from `.env` / environment variables
(see `.env.example` for the full annotated list). Highlights:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model (dimension 384) |
| `EMBEDDING_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Character window + overlap (overlap must be < size) |
| `HYBRID_FUSION` | `rrf` | `rrf` (default) or `weighted` fusion of semantic + BM25 |
| `RERANKER_ENABLED` | `true` | Toggle the cross-encoder reranker |
| `LLM_PROVIDER` | `echo` | `echo` (offline) / `ollama` / `openai` (interface-only) |
| `LLM_BASE_URL` / `LLM_MODEL` | — | e.g. `http://localhost:11434` + `llama3.1` for Ollama |
| `DATA_DIR` | `<project>/data` | Root for ingested sources / sample data / runtime state |
| `LOG_JSON` | `false` | Structured (JSON) logging |

Settings also validate cross-field invariants (e.g. `chunk_overlap < chunk_size`, positive
batch sizes, valid log levels) and derive the vector collection name from the embedding model
so vectors from different models can never be silently mixed.

---

## Project layout

```
src/app/
├── api/                  # FastAPI app + routes + request/response schemas
├── application/services/ # RAGService, RetrievalService, IngestionService, context+prompt, LLMService
├── bootstrap.py          # Composition root (build_container)
├── common/               # config (settings), logging (JSON/redaction), batching, cache, file sources, utils
├── domain/               # entities, value objects, exceptions, and the Protocol interfaces
├── evaluation/           # metrics + runner + `python -m app.evaluation`
├── infrastructure/       # parsers, chunker, embedder, vector store, retrievers, reranker, llm, persistence
└── __main__.py           # CLI entrypoint
data/                     # sample documents + evaluation dataset (runtime state lives here too)
tests/                    # unit + integration suites (96 tests)
scripts/                  # sample data generation
```

Deeper dives:

- [Architecture](docs/architecture.md) — layers, dependency graph, and flow.
- [Ingestion](docs/ingestion.md) — parsing, chunking, embedding, and how indexes stay consistent.
- [Retrieval](docs/retrieval.md) — hybrid fusion, reranking, context assembly, and citations.
- [Evaluation](docs/evaluation.md) — dataset schema, metrics, and how to run/regress.

---

## Testing

```bash
python -m pytest -q          # full suite (96 tests)
python -m pytest tests/unit  -q
python -m pytest tests/integration -q
```

Integration tests run against real Chroma (with a fake embedder) and the FastAPI
`TestClient`; they are self-contained and cleaned up after each run.

## Docker

```bash
docker compose up --build    # serves the API at http://localhost:8000
```

See [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

---

## Known limitations

- Chroma's filter builder implements exact-match (`$eq`) and exclusion (`$ne`) filters;
  `SearchFilters.limit_per_document` is declared but not implemented in the search path.
- `QueryLogRow` exists in the persistence schema, but query logging is not yet wired into
  `RAGService`.
- `LLM_PROVIDER=openai` is an interface placeholder that raises a clear `ConfigurationError`
  until an OpenAI-backed LLM is implemented.
- The `echo` LLM is a deterministic placeholder for pipe-testing; it does not answer facts.