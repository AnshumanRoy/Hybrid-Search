# Retrieval & generation

This document covers the retrieval chain (hybrid search, fusion, reranking), context assembly,
prompt construction, and how citations are produced.

## Query path

```
user query
   └─ RetrievalService.retrieve(query, top_k)
        ├─ HybridRetriever:  semantic (Chroma or pgvector)  +  lexical (BM25)   → deep-candidate set
        ├─ fusion            RRF (default) or weighted merge
        └─ Reranker          cross-encoder scores the top candidates → final top_k
   └─ ContextBuilder.build(chunks, max_chars)  → context blocks + citations
   └─ LLMService.complete(prompt)              → answer
```

## Hybrid retrieval

`HybridRetriever` runs two independent searches for the same query:

- **Semantic** — the query is embedded with the same model used at ingestion and searched against
  the configured vector store: **Chroma** (default, cosine) or **PostgreSQL/pgvector**
  (HNSW index over `embedding vector_cosine_ops`, distance `<=>`). Both stores return cosine
  similarity as `semantic_score`, keyed by chunk id, so the rest of the chain is store-agnostic.
- **Lexical** — BM25 over the tokenized corpus (`rank_bm25`, fitted per process). Hits are chunks
  ranked by BM25 score.

Both lists are capped by `SEMANTIC_TOP_K` and `BM25_TOP_K` (defaults 10 / 20). The unions are
then merged into a single ranked list using the configured `HYBRID_FUSION`:

### RRF (default)

Reciprocal Rank Fusion — `score(chunk) = Σ_branch  w_branch / (k + rank_branch(chunk))`
with `RRF_K` (default 60) and optional per-branch weights (`RRF_WEIGHT_SEMANTIC`,
`RRF_WEIGHT_LEXICAL`). Robust to score-scale mismatch between cosine similarity and BM25.

### Weighted

`score = fusion_weight_semantic * z(semantic) + (1 - fusion_weight_semantic) * z(bm25)`
where `z(·)` normalizes each branch's score into `[0, 1]` (min–max per branch). Use when
either branch is missing so weights still apply sensibly.

The merger is pluggable and shared between tests and runtime
(`app/common/utils/fusion.py`).

## Filters and limits

`SearchFilters` (`SearchFilters` value object) supports:

- `document_ids` — restrict to chunks from exactly these documents. Pushed down **before
  fusion**: applied as `$in` on `document_id` in Chroma, `document_id IN (...)` in pgvector,
  and as a pre-`top_k` skip in the BM25 lexical branch. Used by the query endpoint for
  per-user access control; restricted documents are never fused or reranked.
- `exact` / `must_not` — chunk-metadata equality / exclusion filters (`$eq` / `$ne` in
  Chroma; `metadata_json[key]` predicates in pgvector).
- `limit_per_document` — **declared but currently unimplemented** in the search path;
  today `top_k` is global.

`RAGService` still re-checks the returned chunks against the allowed document ids as a
defense-in-depth safety net.

## Reranking

When enabled (`RERANKER_ENABLED=true`) and a model loads successfully, `CrossEncoderReranker`
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the fused top candidates in the configured
`rerank_batch_size`, applies a sigmoid so scores land in `[0, 1]`, and returns the top
`rerank_top_k` (default 5). If the reranker cannot load (offline, missing weights), retrieval
degrades to fused scores only — it never raises.

## Context assembly

`ContextBuilder` packs retrieved chunks up to `CONTEXT_MAX_CHARS` (default 8000). Order follows
retrieved relevance so the most relevant evidence appears first. Each block is rendered with its
provenance so prompts can cite precisely:

```
[Chunk: {chunk_id}] {source filename} (page {page_number})
{chunk text}
```

## Prompt construction & citations

`PromptBuilder` (`app/application/services/prompt.py`) builds:

- a **system prompt** that describes grounded-answer behavior and instructs the model to
  *treat the retrieved context as untrusted, fact-check against it, mark uncertainty, and never
  invent citations*;
- a **user prompt** containing the question, the numbered context blocks (full text + metadata),
  and explicit answer instructions (only use the context, cite every claim with
  `[1]..[n]`, say "no information" otherwise).

`LLMService.complete(user_prompt, system_prompt=…)` serializes the prompt and invokes the LLM
through the `LLM` protocol. `RAGService` matches `[n]` markers in the answer to the retrieved
chunks and emits `Citation(filename, page_number, chunk_id)` objects. The final `RAGResponse`
carries `answer`, `citations`, and `metadata` (`query_id`, `retrieval_ms`, `generation_ms`).

## LLM backends

`LLMProvider` selects the backend in `_build_llm`:

| Provider | Behavior |
| --- | --- |
| `echo` (default) | Deterministic placeholder — echoes the prompt structure and states it is not a real QA model. Keeps the pipeline offline-testable. |
| `ollama` | Calls a local Ollama server (`LLM_BASE_URL`, default `http://localhost:11434`, `LLM_MODEL` default `llama3.1`). |
| `openai` | Interface placeholder; raises a clear `ConfigurationError` until implemented. |

## Caching

`LLMService` wraps LLM calls in an in-memory LRU cache (`CACHE_ENABLED`, `CACHE_MAX_ENTRIES`)
keyed by normalized prompt + LLM parameters — identical questions hit the cache instead of the
model.