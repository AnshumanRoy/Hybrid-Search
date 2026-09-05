"""Embedder interface conformance + device resolution tests."""

import pytest

from app.domain.exceptions import EmbeddingError
from app.domain.interfaces import Embedder
from app.infrastructure.embedders.sentence_transformers import resolve_device
from tests.fakes import FakeEmbedder


def test_fake_embedder_conforms_to_protocol() -> None:
    embedder: Embedder = FakeEmbedder()
    vectors = embedder.embed(["hello world", "second text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 8


def test_embedding_is_deterministic() -> None:
    e = FakeEmbedder()
    assert e.embed(["same exact text"]) == e.embed(["same exact text"])


def test_embedding_normalized() -> None:
    import math

    e = FakeEmbedder()
    vector = e.embed(["normalize me"])[0]
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6


def test_embed_batch_matches_inputs() -> None:
    e = FakeEmbedder()
    texts = ["a", "b", "c", "d", "e"]
    assert len(e.embed(texts)) == len(texts)


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_valid() -> None:
    assert resolve_device("auto") in ("cpu", "cuda", "mps")


def test_resolve_device_unknown_raises() -> None:
    with pytest.raises(EmbeddingError):
        resolve_device("quantum")


def test_model_properties_exposed() -> None:
    e = FakeEmbedder(model_name="my-model", dimension=16)
    assert e.model_name == "my-model"
    assert e.dimension == 16