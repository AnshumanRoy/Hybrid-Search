"""Chunker unit tests: boundaries, overlap, deterministic ids, edge cases."""

import pytest

from app.domain.entities import DocumentPage, ParsedDocument
from app.infrastructure.chunkers.sentence import SentenceAwareChunker


def _doc(text_by_page: list[str], doc_id: str = "doc_abc") -> ParsedDocument:
    pages = [
        DocumentPage(page_number=i + 1, text=text)
        for i, text in enumerate(text_by_page)
        if text is not None
    ]
    return ParsedDocument(
        document_id=doc_id,
        filename="policy.pdf",
        page_count=len(pages),
        content_hash="abcdef",
        pages=pages,
    )


def test_simple_split_respects_size() -> None:
    text = ("This is the first sentence of the policy document. "
            "It contains a reasonable amount of text to chunk. "
            "This second sentence is longer than the first one. "
            "And this third one provides a little more material.")
    chunks = SentenceAwareChunker(size=60, overlap=0).chunk(_doc([text]))
    assert len(chunks) >= 2
    assert all(len(c.text) <= 180 for c in chunks)


def test_no_sentence_boundary_cut() -> None:
    # 4 sentences, each ~14 chars + terminator => 2 fit in a chunk of ~36.
    s1, s2, s3, s4 = (
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
        "Fourth sentence.",
    )
    text = " ".join([s1, s2, s3, s4])
    chunks = SentenceAwareChunker(size=36, overlap=0).chunk(_doc([text]))
    assert [c.text for c in chunks] == [f"{s1} {s2}", f"{s3} {s4}"]
    # No chunk begins mid-sentence: every chunk starts with a known prefix.
    prefixes = (s1, s2, s3, s4)
    for c in chunks:
        assert any(c.text.startswith(p) for p in prefixes)


def test_deterministic_chunk_ids() -> None:
    text = "Sentence one here. Sentence two there. Sentence three everywhere."
    a = SentenceAwareChunker(size=30, overlap=0).chunk(_doc([text], doc_id="doc_x"))
    b = SentenceAwareChunker(size=30, overlap=0).chunk(_doc([text], doc_id="doc_x"))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.chunk_id.startswith("doc_x") for c in a)
    assert all(c.chunk_index == i for i, c in enumerate(a))


def test_preserves_document_and_page_metadata() -> None:
    text = "This is a single page with a single sentence in it."
    chunks = SentenceAwareChunker(size=200, overlap=0).chunk(_doc([text]))
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_abc"
    assert chunks[0].page_number == 1
    assert chunks[0].filename == "policy.pdf"


def test_overlap_carries_prefix() -> None:
    text = "First sentence of the paragraph. Second sentence in the paragraph. Third sentence remaining."
    chunks = SentenceAwareChunker(size=40, overlap=15).chunk(_doc([text]))
    if len(chunks) >= 2:
        tail = chunks[0].text[-15:]
        assert chunks[1].text.startswith(tail)


def test_empty_pages_yield_no_chunks() -> None:
    chunks = SentenceAwareChunker(size=40, overlap=10).chunk(_doc(["", None]))
    assert chunks == []


def test_multiple_pages_chunk_index_globally_increments() -> None:
    doc = _doc(["First page sentence. Another here.", "Second page content."])
    chunks = SentenceAwareChunker(size=30, overlap=0).chunk(doc)
    assert len(chunks) >= 2
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))
    assert chunks[-1].page_number == 2


def test_oversized_single_sentence_is_split() -> None:
    long_sentence = " ".join(["word"] * 500) + " final."
    chunks = SentenceAwareChunker(size=100, overlap=0).chunk(_doc([long_sentence]))
    assert chunks
    assert max(len(c.text) for c in chunks) <= 110


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceAwareChunker(size=50, overlap=60)