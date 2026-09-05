"""Well-typed domain exceptions.

Hierarchy::

    RAGError (BaseException)
      ├── ConfigurationError
      ├── DocumentSourceError
      ├── DocumentParseError   ── UnsupportedDocumentError
      ├── EmbeddingError
      ├── VectorStoreError
      ├── RetrievalError
      ├── RerankError
      ├── LLMError
      └── RepositoryError
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all application-defined, expected failures."""


class ConfigurationError(RAGError):
    """Invalid or missing configuration."""


class DocumentSourceError(RAGError):
    """Failure reading from a file/object source (not found, unreadable, path traversal)."""


class DocumentParseError(RAGError):
    """Failure extracting text from an uploaded document."""


class UnsupportedDocumentError(DocumentParseError):
    """The file extension/type is not supported by any registered parser."""


class EmbeddingError(RAGError):
    """Failure producing or loading embeddings."""


class VectorStoreError(RAGError):
    """Failure interacting with the vector store."""


class RetrievalError(RAGError):
    """Failure during query retrieval or fusion."""


class RerankError(RAGError):
    """Failure during query-document reranking."""


class LLMError(RAGError):
    """Failure generating a response (timeout, provider error, bad output)."""


class RepositoryError(RAGError):
    """Failure persisting or loading records from the repository layer."""