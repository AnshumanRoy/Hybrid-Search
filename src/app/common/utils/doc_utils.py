"""Document-level utilities: content hashing, size/extension validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from app.common.configuration.settings import Settings
from app.common.file_sources.local import LocalFileSource
from app.domain.entities import FileReference
from app.domain.exceptions import DocumentSourceError, UnsupportedDocumentError

CHUNK_HASH_SEPARATOR = "\n--\n"


def content_hash(data: bytes) -> str:
    """Stable SHA-256 fingerprint of raw document bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_chunk_id(document_id: str, page_number: int | None, chunk_index: int) -> str:
    """Deterministic chunk id so re-runs produce stable references.

    Format: ``<document_id>::p<page>::<index>``.  Stable across processes and
    chunker parameter changes for the same document/page/index position.
    """
    page = f"p{page_number}" if page_number is not None else "px"
    return f"{document_id}::{page}::i{chunk_index}"


def normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def assert_supported(filename: str, ext: Sequence[str]) -> str:
    inner = normalize_extension(filename)
    if not inner:
        raise UnsupportedDocumentError(f"cannot infer extension for '{filename}'")
    if inner not in [e.lstrip(".").lower() for e in ext]:
        raise UnsupportedDocumentError(
            f"unsupported document type '.{inner}' (allowed: {', '.join(sorted(ext))})"
        )
    return inner


def assert_size_ok(file: FileReference, max_bytes: int) -> None:
    size = file.size_bytes
    if size is None:
        return
    if size <= 0:
        raise DocumentSourceError(f"file is empty: {file.path}")
    if size > max_bytes:
        raise DocumentSourceError(
            f"file exceeds maximum size: {file.path} ({size} > {max_bytes} bytes)"
        )


def load_source(settings: Settings) -> LocalFileSource:
    return LocalFileSource(settings.resolve_data_dir(), extensions=settings.allowed_extensions)