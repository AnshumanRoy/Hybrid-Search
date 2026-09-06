"""Deterministic query rewriter — no LLM call.

Typo/stopword normalization plus a small synonym-expansion table. Every output
invariant is reproducible and side-effect free, which makes it the default
fallback for ``RewriteService`` and the strategy of choice in tests.
"""

from __future__ import annotations

import re

from app.domain.value_objects import MAX_REWRITE_VARIANTS, RewrittenQuery

_CONTRACTIONS = {
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "dont": "do not",
    "don't": "do not",
    "doesnt": "does not",
    "doesn't": "does not",
    "cant": "cannot",
    "im": "i am",
    "i'm": "i am",
    "it's": "it is",
    "thats": "that is",
    "that's": "that is",
}

_COMMON_TYPOS = {
    "hte": "the",
    "teh": "the",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
    "occured": "occurred",
    "becuase": "because",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "being", "how", "what", "which", "who", "whom",
    "whose", "when", "where", "why", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "i", "you", "we", "they", "he",
    "she", "it", "me", "him", "her", "us", "them", "my", "your", "our", "their", "its",
    "this", "that", "these", "those", "with", "about", "from", "as", "by", "but", "if",
    "than", "then", "so", "very", "just", "also", "too", "not", "no", "yes", "more",
    "most", "such", "each", "other", "another", "any", "all", "some",
}

_SYNONYMS = {
    "refund": "refunds",
    "refunds": "refund",
    "return": "returns",
    "returns": "return",
    "policy": "policies",
    "policies": "policy",
    "price": "pricing",
    "pricing": "price",
    "pay": "payment",
    "payment": "pay",
    "shipping": "delivery",
    "delivery": "shipping",
    "cancel": "cancellation",
    "cancellation": "cancel",
}


class HeuristicQueryRewriter:
    """Normalization + synonym variants, capped at ``max_variants``."""

    def __init__(self, max_variants_cap: int = MAX_REWRITE_VARIANTS) -> None:
        self.max_variants_cap = max_variants_cap

    def rewrite(
        self,
        query: str,
        *,
        max_variants: int,
        context_hint: str | None = None,
    ) -> RewrittenQuery:
        original = query.strip()
        count = max(1, min(max_variants, self.max_variants_cap))

        variants = [original]
        normalized = self._normalize(original)
        if normalized and normalized != original:
            variants.append(normalized)
        for candidate in self._synonym_variants(original):
            if len(variants) >= count:
                break
            if candidate not in variants:
                variants.append(candidate)

        return RewrittenQuery(
            original=original,
            variants=variants[:count],
            strategy="heuristic",
        )

    def _normalize(self, text: str) -> str:
        expanded = text
        for contraction, full in _CONTRACTIONS.items():
            expanded = re.sub(rf"\b{re.escape(contraction)}\b", full, expanded, flags=re.IGNORECASE)
        words: list[str] = []
        for raw in re.findall(r"[a-zA-Z0-9]+", expanded.lower()):
            word = _COMMON_TYPOS.get(raw, raw)
            if word in _STOPWORDS:
                continue
            words.append(word)
        return " ".join(words)

    def _synonym_variants(self, text: str) -> list[str]:
        words = text.lower().split()
        variants: list[str] = []
        for i, word in enumerate(words):
            replacement = _SYNONYMS.get(word)
            if replacement is None:
                continue
            variants.append(" ".join(words[:i] + [replacement] + words[i + 1 :]))
        return variants