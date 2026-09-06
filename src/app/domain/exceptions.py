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
      ├── LLMError ── LLMAuthenticationError
      ├── UnknownModelError
      ── RepositoryError
      └── AuthenticationError
      └── PermissionDeniedError
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


class LLMAuthenticationError(LLMError):
    """The LLM provider rejected our credentials (e.g. NIM 401/403).

    Kept distinct from ``LLMError`` so the API can return a specific status
    (401) instead of a generic 500 when the configured provider key is bad.
    """


class UnknownModelError(RAGError):
    """The requested model is not registered in the LLM provider registry."""

    def __init__(self, model_name: str | None) -> None:
        self.model_name = model_name
        super().__init__(f"unknown model: {model_name or '<none>'}")


class RepositoryError(RAGError):
    """Failure persisting or loading records from the repository layer."""


class DocumentNotFoundError(RAGError):
    """The requested document does not exist (or is not accessible)."""


class AuthenticationError(RAGError):
    """Invalid credentials or an invalid/expired authentication token."""


class InvalidCredentialsError(AuthenticationError):
    """Wrong email/password on login."""


class UserAlreadyExistsError(RAGError):
    """A user with the given email is already registered."""


class UserNotFoundError(RAGError):
    """A user with the given email does not exist."""


class PermissionDeniedError(RAGError):
    """The authenticated user lacks the required permission on a resource."""


class RateLimitExceededError(RAGError):
    """The authenticated user exceeded a per-user request/usage quota."""