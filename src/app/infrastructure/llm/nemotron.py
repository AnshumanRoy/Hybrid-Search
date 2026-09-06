"""NVIDIA NIM via the OpenAI-compatible ``/chat/completions`` protocol.

Architecture
------------
``NIMClientBase`` is an abstract, provider-agnostic chat-completions client:
it owns transport, payload building, retry-with-backoff, response parsing and
error normalization (raising only ``LLMError``/``LLMAuthenticationError`` /
``ConfigurationError`` — never raw transport exceptions).

Concrete model classes subclass it and describe themselves with class
attributes (``model_id``, ``default_base_url``); they may override the small
hook methods (``_payload_extras``, ``_request_headers``, ``_extract_content``,
``_error_message``) for provider-specific behaviour. ``NemotronNIMClient`` is
such a model (hosted or self-hosted NIM, thinking-enabled).

Error contract (shared):
- 401/403 → ``LLMAuthenticationError`` (auth) — the API key is never echoed.
- 4xx (other) → ``LLMError`` immediately, no retry.
- 429 / 5xx / transport timeouts → bounded retry with exponential backoff,
  honoring ``Retry-After`` (capped); exhausted retries → ``LLMError``.
"""

from __future__ import annotations

import time
from abc import ABC
from typing import ClassVar

from app.domain.exceptions import ConfigurationError, LLMAuthenticationError, LLMError

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 5.0
_BASE_BACKOFF_SECONDS = 0.5


class NIMClientBase(ABC):
    """OpenAI-compatible ``/chat/completions`` client with retry + parsing.

    To add another model describe it via class attributes and (optionally)
    override one of the ``_...`` hooks::

        class MyNIMModel(NIMClientBase):
            model_id = "nvidia/my-model-1b"          # sent in the ``model`` field
            default_base_url = "https://integrate.api.nvidia.com/v1"
            supports_thinking = False
            provider_name = "NIM"
            # def _payload_extras(self): return {"top_p": 0.95}
            # def _request_headers(self): return {"x-api-key": self._api_key}
    """

    #: Model identifier sent in the ``model`` request field (subclass-required).
    model_id: ClassVar[str] = ""

    #: Default base URL used when the caller passes ``base_url=None``.
    default_base_url: ClassVar[str] = ""

    #: Whether the endpoint supports NIM ``chat_template_kwargs`` thinking.
    supports_thinking: ClassVar[bool] = False

    #: Used only in error/status messages (never in requests).
    provider_name: ClassVar[str] = "NIM"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        enable_thinking: bool | None = None,
        transport: object | None = None,
    ) -> None:
        self.model = model or self.model_id
        if not self.model:
            raise ConfigurationError(
                "a model id is required — set `model_id` on the subclass or pass `model=`"
            )
        if not base_url and not self.default_base_url:
            raise ConfigurationError(
                "a base URL is required — set `default_base_url` on the subclass or pass `base_url=`"
            )
        url = (base_url or self.default_base_url).rstrip("/")
        self.base_url = url[: -len("/chat/completions")] if url.endswith("/chat/completions") else url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._enable_thinking = bool(enable_thinking)
        self._transport = transport
        self._client = None
        self.last_usage: dict | None = None

    @property
    def model_name(self) -> str:
        """Display name used for cache keys + response metadata (default: model id)."""
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise LLMError(f"httpx is not installed; cannot call {self.provider_name}") from exc

        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens)
        headers = {"Content-Type": "application/json", **self._request_headers()}

        client = self._client or httpx.Client(timeout=self._timeout, transport=self._transport)  # type: ignore[arg-type]
        self._client = client
        request_timeout = timeout_seconds if timeout_seconds is not None else self._timeout

        attempts_left = self._max_retries
        last_detail = None
        retry_after: str | None = None
        while attempts_left > 0:
            attempts_left -= 1
            try:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                last_detail = f"{self.provider_name} request timed out (model={self.model})"
                retry_after = None
            except httpx.RequestError as exc:  # noqa: BLE001 - normalize at the boundary
                last_detail = f"{self.provider_name} request failed (model={self.model}): {exc}"
                retry_after = None
            else:
                if response.status_code < 400:
                    self._record_usage(response)
                    return self._extract_content(response)
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise LLMError(self._error_message(response))
                last_detail = self._error_message(response)
                retry_after = response.headers.get("Retry-After")

            if attempts_left > 0:
                time.sleep(self._backoff_seconds(retry_after, self._max_retries - attempts_left))

        raise LLMError(f"{last_detail}; giving up after {self._max_retries} attempt(s)")

    # --- Extension hooks -------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        """Compose the OpenAI-compatible request body (override for custom schemas)."""
        payload = {
            "model": self.model,
            "messages": [],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": False,
        }
        extras = self._payload_extras()
        if extras:
            payload.update(extras)
        if self.supports_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})
        return payload

    def _payload_extras(self) -> dict:
        """Provider-specific body fields (override in subclasses)."""
        return {}

    def _request_headers(self) -> dict:
        """Auth/request headers (override for non-Bearer gateways or key-less)."""
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _record_usage(self, response) -> None:
        """Stash the last response's ``usage`` for cost observability (best-effort)."""
        try:
            usage = response.json().get("usage")
        except ValueError:
            usage = None
        self.last_usage = usage if isinstance(usage, dict) else None

    def _extract_content(self, response) -> str:
        """Pull the assistant text out of a 2xx response (override if needed)."""
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"{self.provider_name} returned an unparseable response (model={self.model})"
            ) from exc
        content = content.strip() if isinstance(content, str) else ""
        if not content:
            raise LLMError(f"{self.provider_name} returned an empty response (model={self.model})")
        return content

    def _error_message(self, response) -> str:
        """Human-readable, key-safe message for a non-2xx response (override if needed)."""
        code = response.status_code
        detail = ""
        try:
            error = response.json().get("error")
        except ValueError:
            error = None
        if isinstance(error, dict):
            detail = error.get("message", "")
        elif isinstance(error, str):
            detail = error
        if self._api_key and detail:
            detail = detail.replace(self._api_key, "[REDACTED]")
        if code in (401, 403):
            raise LLMAuthenticationError(
                f"{self.provider_name} authentication failed (status {code}); check your API key"
            )
        return f"{self.provider_name} request failed (status {code}, model={self.model}): {detail or 'unknown error'}"

    @staticmethod
    def _backoff_seconds(retry_after: str | None, failures: int) -> float:
        """Exponential backoff: 0.5s, 1s, 2s, ... capped at 5s.

        ``failures`` counts completed failed attempts (0 before the first
        retry). An integer ``Retry-After`` header overrides the schedule.
        """
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = None
            if delay is not None:
                return min(delay, _MAX_BACKOFF_SECONDS)
        return min(_BASE_BACKOFF_SECONDS * (2**failures), _MAX_BACKOFF_SECONDS)


class NemotronNIMClient(NIMClientBase):
    """NVIDIA Nemotron 3.5 Lightning hosted on NIM (or self-hosted NIM).

    Requires an API key. Uses NIM's thinking option (off unless enabled).
    """

    model_id: ClassVar[str] = "nvidia/nemotron-3.5-lightning-30b-a3b"
    default_base_url: ClassVar[str] = "https://integrate.api.nvidia.com/v1"
    supports_thinking: ClassVar[bool] = True
    provider_name: ClassVar[str] = "NIM"

    #: Environment variable that supplies the API key (for docs/error text).
    api_key_env_var: ClassVar[str] = "NVIDIA_API_KEY"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        enable_thinking: bool | None = None,
        transport: object | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("NVIDIA_API_KEY is required to call NVIDIA NIM")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            enable_thinking=enable_thinking,
            transport=transport,
        )