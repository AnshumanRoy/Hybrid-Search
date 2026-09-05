"""Context construction: dedupe → budget → consistent-source formatting.

The context builder never calls the LLM; it only formats retrieved chunks into
the exact section structure the prompt builder expects.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.domain.entities import RetrievedChunk

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, max_chars: int = 8000) -> None:
        if max_chars < 100:
            raise ValueError("context_max_chars must be >= 100")
        self.max_chars = max_chars

    def build(self, chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
        """Format the given retrieved chunks into source blocks, bounded by
        ``max_chars`` (falls back to the configured budget when omitted)."""
        budget = max_chars or self.max_chars
        selected = self._select(chunks, budget)
        parts: list[str] = []
        for i, chunk in enumerate(selected, start=1):
            parts.append(self._format_source(i, chunk))
        return "\n\n".join(parts)

    def _select(self, chunks: list[RetrievedChunk], budget: int) -> list[RetrievedChunk]:
        seen: set[str] = set()
        selected: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            block = self._format_source(len(selected) + 1, chunk)
            if used + len(block) > budget and selected:
                break
            selected.append(chunk)
            used += len(block)
        return selected

    @staticmethod
    def _format_source(number: int, chunk: RetrievedChunk) -> str:
        page = chunk.page_number if chunk.page_number is not None else "n/a"
        return (
            f"[Source {number}]\n"
            f"Document: {chunk.filename}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"\n{chunk.text}"
        )