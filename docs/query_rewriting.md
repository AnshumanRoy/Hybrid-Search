# Query rewriting — multi-query expansion (design note)

## Scope

Add **multi-query expansion** to the RAG pipeline: one input query → 1..N retrieval-oriented
variants → retrieve for each → cross-variant fusion → rerank → (optionally) answer. Exposed
standalone on `POST /rewrite` (no retrieval) and wired into `POST /query` behind a settings
flag (default **off**). Stateless: conversation history is the orchestrator's job; this module
accepts `context_hint` for forward compatibility only.

## Components

```
domain:      QueryRewriter (Protocol)           → rewrite(query, *, max_variants, context_hint) -> RewrittenQuery
             RewrittenQuery(original, variants, strategy)   [pydantic value object]
             MAX_REWRITE_VARIANTS = 10          [hard ceiling, domain constant]
infra:       LLMQueryRewriter     — wraps a registered LLM backend via LLMService (backend-agnostic)
             HeuristicQueryRewriter — deterministic normalization/synonym expansion, no LLM call
app:         RewriteService       — strategy selection, max_variants clamping, rewrite-level cache,
                                    fail-open orchestration (primary → heuristic → passthrough)
             RetrievalService.retrieve_multi(RewrittenQuery, ...) — fan-out + cross-variant fusion
delivery:    POST /rewrite  ({query, strategy?, max_variants?})   → RewrittenQuery (no retrieval)
             POST /query    (+ optional rewrite: bool, max_variants overrides)
```

- `QueryRewriter.rewrite()` **always** returns results; it may return passthrough
  (`[original]`) but never raises. `strategy` records what actually ran:
  `"llm"` / `"heuristic"` / `"passthrough"` / `"llm_fallback_heuristic"`.
- `RewrittenQuery.variants` is non-empty, contains `original` first, deduplicated,
  range-clamped to `[1, MAX_REWRITE_VARIANTS]`.

## 1. Fan-out concurrency in `RetrievalService`

**Sequential by default.** A new method shares the existing `retrieve()` pipeline:

```python
def retrieve_multi(self, rewritten: RewrittenQuery, top_k=None, *, filters=None, query_id=None) -> list[RetrievedChunk]:
    # per variant: hybrid_search.retrieve(variant, top_k, filters=filters, query_id=qid)
    # cross-variant merge (below) → reranker.rerank → dedupe
```

- Sequential: deterministic, and avoids assuming thread-safety of the Chroma client and
  `SentenceTransformer` (both are shared singletons; neither documents safe parallel use).
- The whole app is synchronous (sync FastAPI endpoints, `ThreadingSingleFlight` only for cache
  stampede). Adding thread-fan-out is a future flag (`REWRITE_FANOUT_CONCURRENCY`) pending
  thread-safety validation of those two dependencies — noted, not built.
- Every variant gets the **same `filters`** (access-control `document_ids` pushed down before
  fan-out; never varies per variant) and the same `top_k`/`query_id`.

## 2. Cross-variant fusion

Reuse the existing fusion primitive "one level up." Add two pure functions to
`app/common/utils/fusion.py` (leave the existing 2-list functions untouched to avoid
destabilizing `HybridRetriever`):

- `rrf_merge_lists(lists, k, weights=None)` — N-way RRF sum (RRF is rank-based and
  list-length-insensitive, ideal for merging variant rankings).
- `weighted_merge_lists(lists, weights=None)` — N lists, per-list scores min-max normalized
  to [0,1], arbitrary per-list weights.

`RetrievalService.retrieve_multi` builds a `ConfigurableFusion`-shaped merge from the same
settings (`hybrid_fusion`, `rrf_k`, `rrf_weight_*`): mode `rrf` → RRF merge above; mode
`weighted` → variants contribute equally when `hybrid_fusion=weighted` (per-variant weights
are unit — flagged trade-off: per-variant LLM-quality weighting is out of scope). Fused list →
`Reranker` → dedupe (existing `dedupe()`), so rerank still sees one merged candidate pool.

## 3. `RewrittenQuery` and cache key

```python
class RewrittenQuery(BaseModel):
    original: str
    variants: list[str]
    strategy: str            # "llm" | "heuristic" | "passthrough" | "llm_fallback_heuristic"
```

Rewrite cache key (sha256 of `\x00`-joined fields, `rewrite:` prefix):

```text
rewrite:{sha256(query | strategy | max_variants)}
```

- Key includes **only** those three fields — no access filters, no user, no vector-store
  generation: rewriting touches neither documents nor the vector store, so it carries none of
  the leakage/cache-invalidation risk of the vector-store cache.
- `context_hint` is **not** in the key, and therefore must not influence variants in this
  pass (else cache hits would be wrong). `context_hint` is accepted and plumbed but unused for
  generation now; wiring it into the prompt AND the key is future work. (Flagged trade-off —
  this pass keeps the cache key exactly as specified.)
- Backend reused: the existing `Cache` protocol, an **own instance** dedicated to rewriting.
  Backend follows `CACHE_BACKEND` (`memory` default → bounded `InMemoryCache`; `redis` →
  `RedisCache` under the `rewrite:` prefix, shared across replicas with the same fail-open
  contract). Rewrite cache turns off when `CACHE_ENABLED=false`, matching the general caching
  toggle. Redis-backend round-trips (JSON) are reconstructed through `RewrittenQuery` so the
  model invariants are re-checked on every hit. When the rewriter calls `LLMService.generate` it
  passes `cacheable=False` so the mandated rewrite-level cache (with the correct key) is the
  single cache authority.

## 4. Settings flag gate

| Setting | Default | Meaning |
| --- | --- | --- |
| `QUERY_REWRITE_ENABLED` | `false` | Gates the `/query` rewrite step (composition-root wiring, not runtime branching) |
| `QUERY_REWRITE_STRATEGY` | `llm` | `llm` / `heuristic` |
| `REWRITE_MAX_VARIANTS` | `3` | Default variant count; settings-validated to `[1, MAX_REWRITE_VARIANTS]` |
| `REWRITE_LLM_MODEL` | `nemotron` | Registry name used for LLM rewriting (defaults to the task-mandated backend) |
| `REWRITE_TIMEOUT_SECONDS` | `15.0` | Short pre-retrieval budget (see §5) |

**Where the gate lives:** `bootstrap.py` builds `RewriteService` **always** (so `POST /rewrite`
works regardless of the /query flag), but passes it into `RAGService` **only when
`QUERY_REWRITE_ENABLED=true`**. `RAGService.answer()` rewrites iff `self.rewrite_service is
not None`; when disabled the container passes `None`, so `/query` is behaviorally identical to
before the feature (there is no code path that even checks the setting at request time).

Per-request overrides on `/query`: `rewrite: bool | None` (None → the enabled default) and
`max_variants: int | None` (server-clamped to the hard ceiling). If a caller passes
`rewrite=true` while the feature is disabled at the container, the request **silently falls
back to no rewriting** (flagged trade-off; mirrors fail-open philosophy).

`bootstrap.py` container wiring: `RewriteService` is constructed with the primary rewriter +
`HeuristicQueryRewriter` fallback + rewrite cache + settings defaults.

## 5. LLM rewriter + the `timeout_seconds` protocol question

`LLMQueryRewriter` depends only on `LLMService` (the registry wrapper) and a strategy default
model name — it never imports `NemotronNIMClient` or any provider. It builds a
system prompt ("you are a re-ranking/ex-pansion assistant; emit exactly N standalone
variants, one per line, no numbering/quotes/markdown, preserve the original if already
specific") and parses defensively (strip numbering/quotes, drop empties, dedupe, cap, ensure
`original` included first).

**Registry short-timeout (flagged trade-off — needs your call):** the LLM Protocol currently
has no timeout parameter and registered providers carry build-time timeouts (60s for NIM).
To enforce a short rewrite budget without duplicating credentials or instantiating a second
NIM client, propose adding **`timeout_seconds: float | None = None`** to `LLM.generate()`
(and `LLMService.generate` passthrough); httpx-backed providers (Ollama, NIM) honor it
per-request, `EchoLLM` ignores it. Alternative if you prefer to keep the protocol frozen:
build the rewrite backend as a dedicated NIM client instance with the short timeout (violates
"reuse the registry" less cleanly). Default choice: protocol addition.

Fail-open in `RewriteService`: on `LLMError`/timeout/empty output → `HeuristicQueryRewriter`
(`strategy="llm_fallback_heuristic"`); heuristic is pure and cannot raise; a hypothetical
failure there → passthrough `[original]` (`strategy="passthrough"`). Rewrite errors are logged
and never propagate to `/query` or `/rewrite`.

## 6. `POST /rewrite`

- Auth required (`get_current_user`), consistent with every non-health/auth endpoint — it
  draws against shared LLM budget and enables operator attribution.
- Body `{query: str (1..1000), strategy?: "llm"|"heuristic", max_variants?: int}`;
  response `{original, variants, strategy}` (DTO mirrors `RewrittenQuery`).
- No retrieval, no document access, no cache reads of the vector cache. It does touch the
  **rewrite cache**.
- `max_variants` clamped to `[1, MAX_REWRITE_VARIANTS]` server-side — never rejected, never
  honored as-is.

## 7. Tests

- Unit: `HeuristicQueryRewriter` (deterministic expansion, clamping, passthrough-of-original);
  `LLMQueryRewriter` against a stubbed LLM backend (prompt shape, one-variant-per-line parse,
  numbered/quoted input, over-/under-count, empty output → falls to fallback).
- `RewriteService`: strategy selection, request-level clamp, fail-open on `LLMError` from the
  stubbed backend, cache hit/miss on key `(query, strategy, max_variants)`, hint not in key.
- Access-control invariance: reuse access-control fixtures; assert the **exact same
  `SearchFilters`** reached every variant (recording hybrid double) and the post-retrieval
  defense-in-depth filter still runs on merged output — the one place a bug could reintroduce
  leakage.
- `/query` byte-for-byte unchanged when `QUERY_REWRITE_ENABLED=false` (default): recorded
  answers (echo) + retrieved chunk ids identical with and without the feature present.
- `/rewrite` in isolation: no vector-store/embedder interactions (assert retrieval double never
  called).
- Integration: `/query` with rewrite on returns merged results incl. `rewritten_variants`/
  `rewrite_strategy` metadata; `/rewrite` e2e with heuristic strategy.

## 8. Non-goals (unchanged)

No history/session storage; no changes to `Reranker`/`ContextBuilder`/`PromptBuilder`/
vector-store cache; no HyDE or sub-question decomposition (possible future `QueryRewriter`
strategies behind the same Protocol).