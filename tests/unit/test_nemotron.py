"""NemotronNIMClient unit tests — stubbed HTTP via httpx.MockTransport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.domain.exceptions import ConfigurationError, LLMAuthenticationError, LLMError
from app.infrastructure.llm.nemotron import NIMClientBase, NemotronNIMClient

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = "sk-test-1234567890"


def _client(
    handler,
    *,
    max_retries: int = 3,
    enable_thinking: bool = False,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
) -> NemotronNIMClient:
    transport = httpx.MockTransport(handler)
    return NemotronNIMClient(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        enable_thinking=enable_thinking,
        transport=transport,
    )


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "A grounded answer."}}],
            "usage": {"completion_tokens": 5},
        },
    )


def _pending_calls():
    calls: list[httpx.Request] = []
    return calls


def _request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_success_parses_content() -> None:
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok_response()

    result = _client(handler).generate("What is the refund policy?", system_prompt="Be grounded.")
    assert result == "A grounded answer."
    assert len(calls) == 1


def test_request_shape_and_auth_header() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = _request_json(request)
        captured["auth"] = request.headers.get("Authorization")
        return _ok_response()

    client = _client(handler, enable_thinking=False)
    client.generate("Tell me about refunds.", system_prompt="Instructions.")

    assert captured["method"] == "POST"
    assert captured["url"] == f"{BASE_URL}/chat/completions"
    body = captured["body"]
    assert body["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert body["messages"] == [
        {"role": "system", "content": "Instructions."},
        {"role": "user", "content": "Tell me about refunds."},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 512
    assert body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["auth"] == f"Bearer {API_KEY}"


def test_explicit_temperature_and_max_tokens_override_defaults() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _request_json(request)
        return _ok_response()

    _client(handler).generate("q", temperature=0.7, max_tokens=128)
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["max_tokens"] == 128


def test_thinking_enabled_flag() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _request_json(request)
        return _ok_response()

    _client(handler, enable_thinking=True).generate("q")
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_empty_content_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    with pytest.raises(LLMError, match="empty response"):
        _client(handler).generate("q")


def test_malformed_response_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(LLMError, match="unparseable"):
        _client(handler).generate("q")


def test_authentication_error_does_not_echo_key() -> None:
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            401,
            json={"error": {"message": f"invalid credentials {API_KEY}", "type": "auth"}},
        )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        _client(handler).generate("q")
    message = str(exc_info.value)
    assert "authentication failed" in message
    assert API_KEY not in message
    assert len(calls) == 1


def test_bad_request_no_retry() -> None:
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "max_tokens is too large"}})

    with pytest.raises(LLMError) as exc_info:
        _client(handler, max_retries=3).generate("q")
    assert "400" in str(exc_info.value)
    assert len(calls) == 1


def test_rate_limit_then_success_retries() -> None:
    attempts = {"n": 0}
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return _ok_response()

    result = _client(handler, max_retries=3).generate("q")
    assert result == "A grounded answer."
    assert attempts["n"] == 2


def test_retry_after_header_used() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"message": "slow down"}})
        return _ok_response()

    result = _client(handler, max_retries=3).generate("q")
    assert result == "A grounded answer."
    assert attempts["n"] == 2


def test_persistent_server_error_gives_up() -> None:
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(LLMError) as exc_info:
        _client(handler, max_retries=2).generate("q")
    assert "overloaded" in str(exc_info.value)
    assert "giving up after 2 attempt(s)" in str(exc_info.value)
    assert len(calls) == 2


def test_timeout_retries_then_gives_up() -> None:
    calls = _pending_calls()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.TimeoutException("connect timed out")

    with pytest.raises(LLMError) as exc_info:
        _client(handler, max_retries=2).generate("q")
    assert "timed out" in str(exc_info.value)
    assert len(calls) == 2


def test_constructor_requires_key_and_model() -> None:
    with pytest.raises(ConfigurationError):
        NemotronNIMClient(model="m", api_key="")
    # Empty model falls back to the subclass model_id; no model anywhere fails.
    client = NemotronNIMClient(api_key=API_KEY)
    assert client.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_base_url_strips_trailing_chat_completions_path() -> None:
    client = NemotronNIMClient(
        model="m",
        api_key=API_KEY,
        base_url="https://example.com/v1/chat/completions",
    )
    assert client.base_url == "https://example.com/v1"


def test_nim_base_is_extensible_by_model_subclasses() -> None:
    class FakeNIMModel(NIMClientBase):
        model_id = "nvidia/fake-model-1b"
        default_base_url = "https://fake-nim.example.com/v1"
        provider_name = "FakeNIM"

        def _payload_extras(self) -> dict:
            return {"top_p": 0.9}

        def _request_headers(self) -> dict:
            return {"x-api-key": self._api_key}

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _request_json(request)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = FakeNIMModel(api_key="k", transport=httpx.MockTransport(handler))
    assert client.model_name == "nvidia/fake-model-1b"
    assert client.generate("hi") == "ok"
    assert captured["body"]["model"] == "nvidia/fake-model-1b"
    assert captured["body"]["top_p"] == 0.9
    assert captured["headers"]["x-api-key"] == "k"
    assert "authorization" not in {h.lower() for h in captured["headers"]}


def test_nim_base_requires_specification() -> None:
    with pytest.raises(ConfigurationError, match="model id"):
        NIMClientBase(api_key="k", base_url="https://example.com/v1")
    with pytest.raises(ConfigurationError, match="base URL"):
        NIMClientBase(model="m", api_key="k")