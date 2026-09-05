"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.common.configuration.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_model="fake-model",
        vector_dimension=8,
        embedding_batch_size=4,
        vector_store_dir=tmp_path / "vectorstore",
        database_url="sqlite://",
        data_dir=tmp_path / "data",
        chunk_size=120,
        chunk_overlap=20,
        cache_enabled=False,
    )


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def eps() -> float:
    return 1e-6