"""PDF parser tests using real PyMuPDF-generated PDFs (no network needed)."""

import io

import pytest

from app.common.configuration.settings import Settings
from app.domain.entities import FileReference
from app.domain.exceptions import DocumentParseError
from app.infrastructure.document_parsers.pdf import PDFParser


@pytest.fixture
def parser():
    return PDFParser(Settings(max_pdf_pages=50))


def _make_pdf(page_texts: list[str]) -> bytes:
    import fitz  # PyMuPDF

    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _reference(name: str = "sample.pdf") -> FileReference:
    return FileReference(path=name, filename=name, uri=f"file:///{name}", source_id="default")


def test_parses_pages_preserving_order_and_metadata(parser) -> None:
    data = _make_pdf(["First page content.", "Second page content. More here."])
    doc = parser.parse(_reference(), data)
    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "First page content" in doc.pages[0].text
    assert "Second page content" in doc.pages[1].text
    assert doc.content_hash  # str non-empty
    assert doc.metadata.get("page_count") == 2


def test_empty_page_kept_with_empty_flag(parser) -> None:
    data = _make_pdf(["Some text", ""])
    doc = parser.parse(_reference(), data)
    assert doc.page_count == 2
    assert doc.pages[1].text == ""
    assert doc.pages[1].metadata.get("is_empty") is True


def test_non_pdf_bytes_rejected(parser) -> None:
    with pytest.raises(DocumentParseError):
        parser.parse(_reference(), b"this is definitely not a pdf file content")


def test_malformed_pdf_raises_meaningful_error(parser) -> None:
    with pytest.raises(DocumentParseError) as excinfo:
        parser.parse(_reference(), b"%PDF-1.7\nGARBAGENOTAREALPDF")
    assert "PDF" in str(excinfo.value)


def test_deterministic_content_hash(parser) -> None:
    data = _make_pdf(["stable text here"])
    a = parser.parse(_reference(), data)
    b = parser.parse(_reference(), data)
    assert a.content_hash == b.content_hash
    assert a.document_id == b.document_id


def test_max_pages_respected(parser) -> None:
    data = _make_pdf([f"page {i}" for i in range(10)])
    doc = parser.parse(_reference(), data)
    assert doc.page_count <= 50