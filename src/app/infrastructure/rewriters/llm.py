"""LLM-based query rewriter — wraps the multi-backend ``LLM`` registry.

Backend-agnostic by design: depends only on ``LLMService`` and a model *name*;
it never imports ``NemotronNIMClient`` or any provider class. Generation
failures raise ``LLMError``; the fail-open orchestration (heuristic fallback,
passthrough) lives in ``RewriteService``, not here.
"""

from __future__ import annotations

import re

from app.application.services.llm_service import LLMService
from app.domain.value_objects import MAX_REWRITE_VARIANTS, RewrittenQuery

_SYSTEM_PROMPT = (
    "You are a query-expansion assistant for a dense + lexical document retrieval system. "
    "Rewrite the user's single search query into retrieval-oriented variants. Rules:\n"
    "- Each variant is a standalone, self-contained search query (no conversational context).\n"
    "- Cover paraphrases, synonyms, and plausible underspecified or ambiguous angles.\n"
    "- Output exactly the requested number of variants, one per line.\n"
    "- No numbering, bullets, quotes, markdown, or commentary — raw query lines only.\n"
    "- Keep each variant short (200 characters max).\n"
    "- First line should be the original phrasing, unchanged, if it is already specific."
)


class LLMQueryRewriter:
    """Rewrites a query into N variants via a registered ``LLM`` backend."""

    def __init__(
        self,
        llm: LLMService,
        *,
        default_model: str = "nemotron",
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_variant_length: int = 200,
        timeout_seconds: float | None = None,
    ) -> None:
        self.llm = llm
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_variant_length = max_variant_length
        self.timeout_seconds = timeout_seconds

    def rewrite(
        self,
        query: str,
        *,
        max_variants: int,
        context_hint: str | None = None,
    ) -> RewrittenQuery:
        original = query.strip()
        count = max(1, min(max_variants, MAX_REWRITE_VARIANTS))

        response = self.llm.generate(
            self._build_prompt(original, count),
            system_prompt=_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            cacheable=False,
            model=self.default_model,
            timeout_seconds=self.timeout_seconds,
        )

        variants = self._finalize(original, self._parse(response), count)
        return RewrittenQuery(original=original, variants=variants, strategy="llm")

    @staticmethod
    def _build_prompt(query: str, count: int) -> str:
        return (
            f"Original query: {query}\n\n"
            f"Produce {count} retrieval-oriented variant(s) of it, one per line, "
            "following the system rules."
        )

    @staticmethod
    def _parse(response: str) -> list[str]:
        lines: list[str] = []
        for raw in response.splitlines():
            line = raw.strip()
            line = re.sub(r"^\s*[-•*\d]+[.)\]]?\s*", "", line)
            line = line.strip('"\'`').strip()
            line = line.rstrip(". ")
            if not line:
                continue
            if line not in lines:
                lines.append(line)
        return lines

    def _finalize(self, original: str, parsed: list[str], count: int) -> list[str]:
        variants: list[str] = [original]
        for candidate in parsed:
            candidate = candidate.strip()
            if not candidate:
                continue
            candidate = candidate[: self.max_variant_length]
            if candidate in variants:
                continue
            variants.append(candidate)
            if len(variants) >= count:
                break
        return variants