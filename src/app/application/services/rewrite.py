"""Rewrite orchestration: strategy selection, clamping, best-effort cache,
and the fail-open guarantee (LLM failure → heuristic → passthrough).

``RewriteService`` is the only place that turns ``QueryRewriter`` strategies
into a request-level answer; it guarantees the no-raise contract of
``QueryRewriter`` at the orchestration boundary. Concrete rewriters (heuristic
never raises; LLM raises ``LLMError`` on backend failure) stay simple.
"""

from __future__ import annotations

import hashlib
import logging

from app.common.cache import Cache
from app.domain.interfaces import QueryRewriter
from app.domain.value_objects import MAX_REWRITE_VARIANTS, RewrittenQuery

logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("llm", "heuristic")


class RewriteService:
    def __init__(
        self,
        llm_rewriter: QueryRewriter | None,
        heuristic_rewriter: QueryRewriter,
        *,
        cache: Cache | None = None,
        default_strategy: str = "llm",
        default_max_variants: int = 3,
        max_variants_cap: int = MAX_REWRITE_VARIANTS,
    ) -> None:
        self.llm_rewriter = llm_rewriter
        self.heuristic_rewriter = heuristic_rewriter
        self.cache = cache
        self.default_strategy = default_strategy
        self.default_max_variants = default_max_variants
        self.max_variants_cap = max_variants_cap

    def rewrite(
        self,
        query: str,
        *,
        strategy: str | None = None,
        max_variants: int | None = None,
        context_hint: str | None = None,
    ) -> RewrittenQuery:
        """Rewrite ``query`` and return variants — never raises.

        ``strategy``/``max_variants`` override the defaults; ``max_variants``
        is clamped to ``[1, max_variants_cap]`` (never rejected, never honored
        as-is). ``context_hint`` is accepted for forward compatibility but is
        deliberately excluded from the cache key and from variant generation.
        Cache is best-effort: a broken backend degrades to uncached rewriting.
        """
        query = query or ""
        strategy = strategy or self.default_strategy
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"unsupported rewrite strategy: {strategy}")
        count = max(1, min(max_variants or self.default_max_variants, self.max_variants_cap))

        key = self._cache_key(query, strategy, count)
        if self.cache is not None:
            cached = self._cache_get(key)
            if cached is not None:
                logger.debug("rewrite cache hit; strategy=%s variants=%d", strategy, count)
                return cached

        result = self._execute(query, strategy, count, context_hint)
        if self.cache is not None:
            self._cache_set(key, result)
        return result

    def _execute(
        self,
        query: str,
        strategy: str,
        count: int,
        context_hint: str | None,
    ) -> RewrittenQuery:
        if strategy == "heuristic":
            return self.heuristic_rewriter.rewrite(query, max_variants=count, context_hint=context_hint)

        if self.llm_rewriter is None:
            logger.warning("rewrite strategy is 'llm' but no LLM rewriter is registered; using heuristic")
            return self._relabel(
                self.heuristic_rewriter.rewrite(query, max_variants=count, context_hint=context_hint),
                "heuristic",
            )

        try:
            return self.llm_rewriter.rewrite(query, max_variants=count, context_hint=context_hint)
        except Exception as exc:  # noqa: BLE001 - failing open is the design
            logger.warning("LLM rewrite failed (%s); falling back to heuristic strategy", exc)

        try:
            return self._relabel(
                self.heuristic_rewriter.rewrite(query, max_variants=count, context_hint=context_hint),
                "llm_fallback_heuristic",
            )
        except Exception:  # noqa: BLE001 - last resort passthrough
            logger.exception("heuristic rewrite failed; passing the original query through")
            return RewrittenQuery(original=query, variants=[query or ""], strategy="passthrough")

    @staticmethod
    def _relabel(result: RewrittenQuery, strategy: str) -> RewrittenQuery:
        return result.model_copy(update={"strategy": strategy})

    @staticmethod
    def _cache_key(query: str, strategy: str, count: int) -> str:
        payload = "\x00".join([query, strategy, str(count)])
        return f"rewrite:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _cache_get(self, key: str) -> RewrittenQuery | None:
        try:
            cached = self.cache.get(key)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - caching must never break rewriting
            logger.warning("rewrite cache read failed: %s", exc)
            return None
        if isinstance(cached, RewrittenQuery):
            return cached
        # Redis backends serialized the model to JSON and load it back as a
        # dict; reconstruct through the model so its invariants are re-checked.
        if isinstance(cached, dict):
            try:
                return RewrittenQuery(**cached)
            except (TypeError, ValueError):
                return None
        return None

    def _cache_set(self, key: str, result: RewrittenQuery) -> None:
        try:
            self.cache.set(key, result)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("rewrite cache write failed: %s", exc)