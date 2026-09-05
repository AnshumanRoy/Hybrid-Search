"""Chunking — sentence-aware, overlap-padded, deterministic chunk ids."""

from __future__ import annotations

import logging
from typing import Any

from app.common.configuration.settings import Settings
from app.common.utils.doc_utils import compute_chunk_id
from app.domain.entities import Chunk, ParsedDocument
from app.domain.interfaces import Chunker

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = (". ", "! ", "? ", ".\n", "!\n", "?\n", "。 ")


class SentenceAwareChunker(Chunker):
    """Recursive sentence-aware chunker with overlap.

    Strategy per page:
      1. Split page text into paragraphs on blank lines.
      2. Split each paragraph into sentences.
      3. Greedily pack sentences into ``chunk_size`` slots, starting new chunks
         at sentence boundaries (no mid-sentence cuts).
      4. If a single sentence exceeds ``chunk_size``, fall back to character
         slices at word boundaries, then hard splits as a last resort.
      5. Overlap: the last ``overlap`` characters of the previous chunk are
         prepended to the next chunk (dropped when it would duplicate the
         absolute start of the page).
      6. Chunk ids are deterministic for a fixed (document, page, index).
    """

    def __init__(self, size: int, overlap: int) -> None:
        if overlap >= size:
            raise ValueError(f"chunk overlap ({overlap}) must be smaller than chunk size ({size})")
        self.size = size
        self.overlap = overlap

    @classmethod
    def from_settings(cls, settings: Settings) -> "SentenceAwareChunker":
        return cls(size=settings.chunk_size, overlap=settings.chunk_overlap)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page in document.pages:
            page_chunks = self._chunk_page(document, page.page_number, page.text)
            for chunk_text in page_chunks:
                chunks.append(
                    Chunk(
                        chunk_id=compute_chunk_id(document.document_id, page.page_number, index),
                        document_id=document.document_id,
                        filename=document.filename,
                        source_uri=document.source_uri,
                        page_number=page.page_number,
                        text=chunk_text,
                        chunk_index=index,
                        metadata={
                            "page_number": page.page_number,
                            **{k: v for k, v in page.metadata.items() if isinstance(v, (str, int, float, bool))},
                        },
                    )
                )
                index += 1
        return chunks

    def _chunk_page(self, document: ParsedDocument, page_number: int, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        sentences = self._to_sentences(normalized)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            while len(sentence) > self.size:
                # Oversized sentence: flush current buffer, then split the
                # sentence at word boundaries.
                if current:
                    chunks.append(self._build_chunk(" ".join(current), chunks))
                    current, current_len = [], 0
                piece, sentence = self._take_piece(sentence)
                chunks.append(piece)

            if current and current_len + len(sentence) + 1 > self.size:
                combined = " ".join(current)
                chunks.append(self._build_chunk(combined, chunks))
                current, current_len = [], 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            combined = " ".join(current)
            chunks.append(self._build_chunk(combined, chunks))

        return [c for c in chunks if c.strip()]

    def _build_chunk(self, text: str, previous_chunks: list[str]) -> str:
        if not self.overlap or not previous_chunks:
            return text
        tail = previous_chunks[-1][-self.overlap :]
        if not tail or text.startswith(tail):
            return text
        return tail + "\n\n" + text

    def _to_sentences(self, text: str) -> list[str]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        sentences: list[str] = []
        for paragraph in paragraphs:
            sentences.extend(self._split_sentences(paragraph))
        return [s for s in sentences if s]

    def _split_sentences(self, paragraph: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        i = 0
        n = len(paragraph)
        while i < n:
            matched = False
            for term in _SENTENCE_TERMINATORS:
                if paragraph.startswith(term, i):
                    end = i + len(term)
                    pieces.append(paragraph[start:end])
                    start = end
                    i = end
                    matched = True
                    break
            if not matched:
                i += 1
        if start < n:
            pieces.append(paragraph[start:])
        return [p.strip() for p in pieces if p.strip()]

    def _take_piece(self, sentence: str) -> tuple[str, str]:
        """Split an oversized sentence at the last word boundary within size."""
        if len(sentence) <= self.size:
            return sentence, ""
        candidate = sentence[: self.size]
        boundary = max(candidate.rfind(" "), candidate.rfind("\n"))
        if boundary > 0:
            piece, remainder = sentence[:boundary], sentence[boundary + 1 :]
        else:
            piece, remainder = sentence[: self.size], sentence[self.size :]
        return piece.strip(), remainder.strip()