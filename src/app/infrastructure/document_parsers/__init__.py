"""Document parsers.

Framework (PyMuPDF) is imported lazily so the domain and application layers
never import heavy optional framework dependencies.
"""

from app.infrastructure.document_parsers.pdf import PDFParser
from app.infrastructure.document_parsers.text import TextParser
from app.infrastructure.document_parsers.router import DocumentParserRouter

__all__ = ["PDFParser", "TextParser", "DocumentParserRouter"]