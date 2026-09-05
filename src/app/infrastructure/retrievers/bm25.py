"""BM25 lexical retriever (rank-bm25) with incremental index updates.

Metadata/score state is immutable: adding or removing documents never mutates
already-returned results.  The index is rebuilt lazily after mutation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.domain.entities import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever implementing the ``LexicalRetriever`` protocol."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._index_pipeline = None
        self._index = None
        self._corpus: list[Chunk] = []

    def add_documents(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._corpus = list(self._chunks.values())
        self._index_pipeline = None
        self._index = None

    def remove_documents(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._corpus = list(self._chunks.values())
        self._index_pipeline = None
        self._index = None

    def count(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks or not query.strip():
            return []

        if self._index_pipeline is None:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [tokenize(c.text) for c in self._corpus]
            self._index = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
            self._index_pipeline = tokenized_corpus

        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(self._corpus)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievedChunk] = []
        for rank_index in ranked[:top_k]:
            chunk = self._corpus[rank_index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    source_uri=chunk.source_uri,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    lexical_score=float(scores[rank_index]) if scores[rank_index] > 0 else 0.0,
                    retrieval_metadata={"lexical": True, "rank": len(results)},
                )
            )
        return results