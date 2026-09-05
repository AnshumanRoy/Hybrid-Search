"""Configuration tests: validation, defaults, env-based overrides."""

import pytest
from pydantic import ValidationError

from app.common.configuration.settings import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.retrieval_top_k == 10
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100


def test_overlap_must_be_less_than_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=150)


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_metric_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(rrf_k=0)


def test_extensions_normalized() -> None:
    s = Settings(allowed_extensions=[".PDF", "TXT"])
    assert s.allowed_extensions == ["pdf", "txt"]


def test_vector_collection_name_derived_from_model() -> None:
    s = Settings(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert "all-MiniLM-L6-v2" in s.vector_collection_name


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "64")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("EMBEDDING_MODEL", "mocked-override")
    s = Settings()
    assert s.chunk_size == 64
    assert s.chunk_overlap == 10
    assert s.embedding_model == "mocked-override"


def test_model_validator_cross_check() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=50, chunk_overlap=50)