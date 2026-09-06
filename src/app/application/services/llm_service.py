"""LLM orchestration: prompt + system instructions + optional response cache.

Supports a name → provider registry so ``/query`` can select a backend by
model name without any protocol-level changes. Keeps full backward
compatibility with the single-provider form (``LLMService(llm, cache, ...)``).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Mapping

from app.common.cache import Cache
from app.domain.exceptions import UnknownModelError
from app.domain.interfaces import LLM

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        llm: LLM,
        cache: Cache | None = None,
        default_temperature: float = 0.2,
        default_max_tokens: int = 512,
        providers: Mapping[str, LLM] | None = None,
        default_model: str | None = None,
    ) -> None:
        """Single-provider mode uses ``llm``; registry mode uses ``providers``.

        ``providers`` maps a model name to an ``LLM``; when given, ``llm`` is
        still kept as the default provider (usually ``providers[default_model]``)
        for convenience/observability. ``default_model`` selects the backend used
        when a request omits ``model``.
        """
        self.llm = llm
        self.cache = cache
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.providers = providers
        self.default_model = default_model
        self.last_usage: dict | None = None

    def model_name_for(self, model: str | None = None) -> str:
        """Resolve the provider for ``model`` and return its ``model_name``.

        Raises ``UnknownModelError`` for names outside the registry. Called
        before retrieval so bad model names fail fast.
        """
        return self._provider(model).model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        cacheable: bool = True,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("refusing to call the LLM with an empty prompt")

        provider = self._provider(model)
        key = self._cache_key(provider, prompt, system_prompt, temperature, max_tokens, timeout_seconds)
        if self.cache is not None and cacheable:
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug("LLM response cache hit")
                return cached

        response = provider.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            timeout_seconds=timeout_seconds,
        )
        self.last_usage = getattr(provider, "last_usage", None)

        if self.cache is not None and cacheable:
            self.cache.set(key, response)
        return response

    def _provider(self, model: str | None) -> LLM:
        if self.providers is not None:
            name = model if model is not None else self.default_model
            provider = self.providers.get(name) if name is not None else None
            if provider is None:
                raise UnknownModelError(name or self.default_model or "<default>")
            return provider
        if model is not None and model != self.llm.model_name:
            raise UnknownModelError(model)
        return self.llm

    def _cache_key(self, provider: LLM, prompt: str, system: str | None, temperature: float | None, max_tokens: int | None, timeout_seconds: float | None = None) -> str:
        payload = "\x00".join(
            [
                provider.model_name,
                system or "",
                prompt,
                f"{temperature if temperature is not None else self.default_temperature}",
                f"{max_tokens if max_tokens is not None else self.default_max_tokens}",
                f"{timeout_seconds if timeout_seconds is not None else ''}",
            ]
        )
        return f"llm:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"