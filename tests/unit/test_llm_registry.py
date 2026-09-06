"""LLMService provider-registry: model-name resolution, defaults, caches."""

from __future__ import annotations

import pytest

from app.application.services.llm_service import LLMService
from app.domain.exceptions import UnknownModelError
from app.infrastructure.llm.echo import EchoLLM
from tests.fakes import FakeCache, RecordingLLM


def _service(providers: dict, default_model: str, cache: FakeCache | None = None) -> LLMService:
    return LLMService(
        llm=providers[default_model],
        cache=cache,
        providers=providers,
        default_model=default_model,
    )


def test_default_model_used_when_omitted() -> None:
    providers = {"echo": RecordingLLM("echo"), "rec": RecordingLLM("rec")}
    service = _service(providers, "echo")
    assert service.generate("q").startswith("echo:")


def test_explicit_model_selected() -> None:
    providers = {"echo": EchoLLM(), "rec": RecordingLLM("rec")}
    service = _service(providers, "echo")
    assert service.generate("q", model="rec").startswith("rec:")


def test_unknown_model_raises() -> None:
    providers = {"echo": EchoLLM(), "rec": RecordingLLM("rec")}
    service = _service(providers, "echo")
    with pytest.raises(UnknownModelError, match="unknown model: nope"):
        service.generate("q", model="nope")


def test_missing_default_model_raises() -> None:
    service = LLMService(llm=RecordingLLM("rec"), providers={"rec": RecordingLLM("rec")})
    with pytest.raises(UnknownModelError):
        service.generate("q")


def test_model_name_for_resolves_default_and_explicit() -> None:
    providers = {"echo": EchoLLM(), "nemotron": RecordingLLM("nemotron")}
    service = _service(providers, "echo")
    assert service.model_name_for() == "echo"
    assert service.model_name_for("nemotron") == "nemotron"


def test_single_provider_backward_compat() -> None:
    service = LLMService(EchoLLM())
    assert service.generate("q").startswith("[echo]")
    assert service.generate("q", model="echo").startswith("[echo]")
    with pytest.raises(UnknownModelError, match="unknown model: ollama"):
        service.generate("q", model="ollama")


def test_cache_key_namespaced_per_model() -> None:
    echo = RecordingLLM("echo")
    nemotron = RecordingLLM("nemotron")
    cache = FakeCache()
    service = _service({"echo": echo, "nemotron": nemotron}, "echo", cache=cache)

    service.generate("same prompt", model="echo")
    service.generate("same prompt", model="echo")
    service.generate("same prompt", model="nemotron")

    assert len(echo.calls) == 1      # second echo request served from cache
    assert len(nemotron.calls) == 1  # separate key per model → separate cache line
    assert cache.gets == 3


def test_default_temperature_and_max_tokens_forwarded() -> None:
    provider = RecordingLLM("rec")
    service = LLMService(provider, default_temperature=0.3, default_max_tokens=64)
    service.generate("q")
    assert provider.calls[0][2] == 0.3
    assert provider.calls[0][3] == 64