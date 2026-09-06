# Hybrid Search — Production RAG Pipeline

A modular, production-oriented **retrieval-augmented generation (RAG)** pipeline built in Python.
It ingests PDFs/plain-text files, splits them into sentence-aware chunks, and serves groundable
answers backed by **hybrid retrieval** (dense embeddings + BM25 lexical) with optional
**cross-encoder reranking**, structured **citations**, an evaluation harness, a FastAPI server,
a CLI, and Docker packaging.

```
documents → parse → chunk → embed → SQL store + vector store + BM25
                                      ↓
 query ─→ hybrid retrieval ─→ rerank ─→ context/prompt ─→ LLM ─→ answer + citations
```

Most API endpoints require a user account (see [access control](docs/access_control.md)).

---

## Features

- **Layered, dependency-inverted architecture** — services depend on `Protocol` interfaces
  (`app/domain/interfaces.py`); only the composition root (`app/bootstrap.py`) touches frameworks.
- **Multi-format ingestion** — PDF (PyMuPDF) and plain text (`.txt`/`.md`), routed by extension,
  with strict size/extension guards and watch-for-traversal file sources.
- **Sentence-aware chunking** — configurable window and overlap so sentence context is never
  sliced mid-sentence.
- **Three-way index** — SQLite/PostgreSQL (SQLAlchemy) for document/chunk metadata, a vector
  store for dense embeddings (**Chroma by default, or PostgreSQL + pgvector**), and BM25 for
  lexical scoring. Ingest is **idempotent and incremental** via content hashes: unchanged files
  are skipped, changed files are re-indexed atomically.
- **Hybrid retrieval** — merge semantic + BM25 scores with configurable **RRF** (default) or
  **weighted** fusion; optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker. The semantic leg
  runs on Chroma (default) or PostgreSQL/pgvector (HNSW cosine index).
- **Grounded generation** — context assembly with explicit citations (filename, page, chunk),
  deterministic prompt templates that treat retrieved context as *untrusted*.
- **Pluggable LLM backends** — `echo` (offline placeholder), local `ollama`, or NVIDIA
  **Nemotron 3.5 Lightning** via NVIDIA NIM (OpenAI-compatible), selectable per request via
  `model` on `/query` or the CLI `--model` flag. NIM clients share an abstract base so new
  models are thin subclasses.
- **Multi-user documents** — user accounts (password hashing + signed bearer tokens, stdlib),
  per-document ownership, **viewer/editor sharing**, document-level CRUD, query scoping
  so answers only cite documents you can see, token revocation, and per-user rate limits.
- **Evaluation harness** — retrieval metrics (recall/precision@K, MRR, nDCG) plus generation
  metrics (faithfulness, answer/context relevance), with rewrite on/off A/B and latency/cost
  telemetry. 261 tests included (unit + integration).
- **API + CLI** — FastAPI (`/health`, `/query`, `/documents/ingest*`, `/documents`) and a CLI for
  `ingest`, `query`, and `evaluate`.
- **Fail-soft design** — missing/offline reranker, LLM, or embedding model degrades gracefully
  (never crashes the pipeline), with an `echo` LLM for offline smoke tests.

---

## Quickstart

> Python 3.12+ required. On Windows use the provided venv: `env\Scripts\python.exe`.

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure (optional). Copy .env.example -> .env and adjust.
cp .env.example .env

# 3. Ingest the bundled sample documents (acme_handbook.pdf + .txt)
python -m app ingest

# 4. Ask a question (default LLM_PROVIDER=echo prints a deterministic placeholder)
python -m app query "What is Acme's refund window for physical products?"

# 4b. (Optional) Ask through NVIDIA Nemotron 3.5 Lightning via NIM.
#     Requires NVIDIA_API_KEY in .env and NEMOTRON_ENABLED=true (or LLM_PROVIDER=nemotron).
python -m app query "What is Acme's refund window for physical products?" --model nemotron

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
| `python -m app query "question" [--top-k N] [--model NAME]` | Run the full RAG pipeline and print the answer with citations and timing metadata. `--model` selects the LLM backend (`echo`/`ollama`/`nemotron`); defaults to `LLM_PROVIDER`; unknown names exit 1. |
| `python -m app evaluate [dataset] [--generate] [--top-k N]` | Run retrieval (and optional generation) evaluation metrics against a dataset. Use `python -m app.evaluation --rewrite --max-variants N --model NAME` to benchmark multi-query expansion, latency, and token cost. |

## API

All endpoints except `/health`, `/auth/register`, and `/auth/login` require
`Authorization: Bearer <token>` (including `/auth/me`).

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/health` | — | `{status, vector_store{count, backend, ...}, services{...}}` |
| POST | `/auth/register` | `{"email","password","display_name"?}` | `{token, user}` |
| POST | `/auth/login` | `{"email","password"}` | `{token, user}` |
| GET | `/auth/me` | — | `{user_id, email, display_name, created_at}` |
| POST | `/auth/refresh` | — | `{token, user}` — new token, same `token_version`; invalidates nothing |
| POST | `/auth/revoke` | — | `204` — bumps your `token_version`; every outstanding token stops working |
| POST | `/query` | `{"query", "top_k", "model", "rewrite"?}` | `{answer, citations[], retrieved_chunks[], metadata{query_id, llm_model, retrieval_ms, generation_ms, rewritten_variants?, rewrite_strategy?}}` |
| POST | `/rewrite` | `{"query", "strategy"?, "max_variants"?}` (auth required) | `{original, variants[], strategy}` |
| POST | `/documents/ingest` | multipart `file` | `list[{document_id, filename, status, chunks_embedded, error, duration_ms}]` |
| POST | `/documents/ingest-dir` | — | same as above (server-side directory ingest, owned by caller) |
| GET | `/documents` | — | `list[{..., permission:"owner"\|"editor"\|"viewer", shared, owner_email}]` (your scope only) |
| GET | `/documents/{id}` | — | single document with your permission |
| PUT | `/documents/{id}` | multipart `file` | re-index (owner or editor); keeps id + shares |
| DELETE | `/documents/{id}` | — | delete (owner only) |
| POST | `/documents/{id}/share` | `{"user_email","permission":"viewer"\|"editor"}` | grant access (owner only) |
| GET | `/documents/{id}/shares` | — | list shares (owner only) |
| DELETE | `/documents/{id}/share/{email}` | — | revoke access (owner only) |

Interactive docs at `/docs` (Swagger UI). Full details in [docs/access_control.md](docs/access_control.md).

`POST /query` accepts an optional `model` field selecting the LLM backend (`echo`, `ollama`, or
`nemotron`). The registry always contains `echo` and `ollama`; `nemotron` is registered only
when `NEMOTRON_ENABLED=true` or `LLM_PROVIDER=nemotron`. Unknown model names are rejected with
HTTP 404; invalid NVIDIA credentials return HTTP 401. The resolved model is echoed in
`metadata.llm_model`.

**Query rewriting (multi-query expansion).** The orchestrator-facing feature turns one query
into 1–N retrieval variants, retrieves each variant, and cross-variant fuses the results one
level above the existing hybrid fusion. It lives behind `QUERY_REWRITE_ENABLED` (default
`false`) — the standalone `POST /rewrite` endpoint works regardless of the flag, so an
orchestrator can "see" the variants before deciding whether to retrieve. Strategy `llm`
(registry backend, default `nemotron`) or deterministic `heuristic` (typo/stopword
normalization + synonyms); LLM failures fail open to heuristic, then passthrough. `max_variants`
is clamped server-side to `[1, 10]`. See [docs/query_rewriting.md](docs/query_rewriting.md).

---

## Configuration

Settings are typed via pydantic-settings and read from `.env` / environment variables
(see `.env.example` for the full annotated list). Highlights:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VECTOR_STORE_BACKEND` | `chroma` | `chroma` (local directory) or `pgvector` (PostgreSQL — requires `DATABASE_URL` to be a Postgres URL) |
| `DATABASE_URL` | `sqlite:///.../rag.db` | Document store; with pgvector this also holds the vectors |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model (dimension 384) |
| `EMBEDDING_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Character window + overlap (overlap must be < size) |
| `HYBRID_FUSION` | `rrf` | `rrf` (default) or `weighted` fusion of semantic + BM25 |
| `RERANKER_ENABLED` | `true` | Toggle the cross-encoder reranker |
| `LLM_PROVIDER` | `echo` | `echo` (offline) / `ollama` / `nemotron` (NVIDIA NIM) / `openai` (interface-only) |
| `LLM_BASE_URL` / `LLM_MODEL` | — | e.g. `http://localhost:11434` + `llama3.1` for Ollama |
| `NVIDIA_API_KEY` | — | Required for `nemotron` (NVIDIA NIM); set in `.env`, never logged or echoed |
| `NEMOTRON_ENABLED` | `false` | Register `nemotron` for per-request selection without making it the default |
| `NEMOTRON_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM OpenAI-compatible endpoint |
| `NEMOTRON_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Nemotron 3.5 Lightning model served by NIM |
| `NEMOTRON_MAX_TOKENS` | `512` | Max generated tokens |
| `NEMOTRON_TIMEOUT_SECONDS` | `60.0` | HTTP timeout per NIM request |
| `NEMOTRON_MAX_RETRIES` | `3` | Total attempts; exponential backoff, retries timeouts and 429/5xx |
| `NEMOTRON_ENABLE_THINKING` | `false` | Sends `chat_template_kwargs.enable_thinking` to NIM |
| `QUERY_REWRITE_ENABLED` | `false` | Gate automatic rewriting inside `/query` (`/rewrite` always works) |
| `QUERY_REWRITE_STRATEGY` | `llm` | `llm` or `heuristic` — default rewriter for `/query` and `/rewrite` |
| `REWRITE_MAX_VARIANTS` | `3` | Default variant count (clamped to `MAX_REWRITE_VARIANTS=10`) |
| `REWRITE_LLM_MODEL` | `nemotron` | LLM backend used by the `llm` strategy |
| `REWRITE_TIMEOUT_SECONDS` | `15.0` | Per-request timeout for the rewrite LLM call |
| `DATA_DIR` | `<project>/data` | Root for ingested sources / sample data / runtime state |
| `AUTH_SECRET_KEY` | `dev-only-change-me` | Token signing secret — set a long random value in deployment |
| `AUTH_TOKEN_TTL_SECONDS` | `900` | Access-token lifetime (seconds); re-issue via `/auth/refresh` |
| `RATE_LIMIT_ENABLED` | `true` | Gate the per-user rate limiter on `/query` + `/rewrite` |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Burst window length |
| `RATE_LIMIT_MAX_REQUESTS` | `120` | Max requests per user per burst window |
| `RATE_LIMIT_DAILY_MAX_REQUESTS` | `2000` | Max requests per user per day |
| `LOG_JSON` | `false` | Structured (JSON) logging |

Settings also validate cross-field invariants (e.g. `chunk_overlap < chunk_size`, positive
batch sizes, valid log levels) and derive the vector collection name from the embedding model
so vectors from different models can never be silently mixed.

---

## Project layout

```
src/app/
├── api/                  # FastAPI app + routes + request/response schemas
├── application/services/ # RAGService, RetrievalService, IngestionService, RewriteService, context+prompt, LLMService
├── bootstrap.py          # Composition root (build_container)
├── common/               # config (settings), logging (JSON/redaction), batching, cache, file sources, utils
├── domain/               # entities, value objects, exceptions, and the Protocol interfaces
├── evaluation/           # metrics + runner + `python -m app.evaluation`
├── infrastructure/       # parsers, chunker, embedder, vector store, retrievers, reranker, llm, rewriters, persistence
└── __main__.py           # CLI entrypoint
data/                     # sample documents + evaluation dataset (runtime state lives here too)
tests/                    # unit + integration suites (261 tests)
scripts/                  # sample data generation
```

Deeper dives:

- [Architecture](docs/architecture.md) — layers, dependency graph, and flow.
- [Ingestion](docs/ingestion.md) — parsing, chunking, embedding, and how indexes stay consistent.
- [Retrieval](docs/retrieval.md) — hybrid fusion, reranking, context assembly, and citations.
- [Query rewriting](docs/query_rewriting.md) — multi-query expansion design, strategies, and the fail-open contract.
- [Evaluation](docs/evaluation.md) — dataset schema, metrics, and how to run/regress.
- [Access control](docs/access_control.md) — accounts, ownership, sharing, and query scoping.

---

## Testing

```bash
python -m pytest -q          # full suite (261 tests)
python -m pytest tests/unit  -q
python -m pytest tests/integration -q
```

Integration tests run against real Chroma (with a fake embedder) and the FastAPI
`TestClient`; they are self-contained and cleaned up after each run. The API
tests exercise the full auth + ownership + sharing flows over HTTP.

## Docker

```bash
docker compose up --build        # PostgreSQL(pgvector) + API; serves at http://localhost:8000
docker compose up postgres       # Postgres only — handy for a local DEV test of the pgvector backend
```

The compose stack runs the API with `VECTOR_STORE_BACKEND=pgvector` against the bundled
PostgreSQL service. To switch back to Chroma or a different Postgres, override the environment
variables (see `.env.example`).

See [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

---

## Known limitations

- Chroma's filter builder implements exact-match (`$eq`), exclusion (`$ne`), and document
  scoping (`$in` on `document_id`); pgvector implements the same predicates, plus
  `document_id IN (...)`. `SearchFilters.limit_per_document` is declared but not implemented
  in the search path.
- Query scoping (`SearchFilters.document_ids`) is pushed down into both the vector store and
  the BM25 index **before** fusion/reranking; the application-layer re-check is a
  defense-in-depth safety net.
- `QueryLogRow` exists in the persistence schema, but query logging is not yet wired into
  `RAGService`.
- `LLM_PROVIDER=openai` is an interface placeholder that raises a clear `ConfigurationError`
  until an OpenAI-backed LLM is implemented.
- The `echo` LLM is a deterministic placeholder for pipe-testing; it does not answer facts.
