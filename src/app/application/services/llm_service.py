"""LLM orchestration: prompt + system instructions + optional response cache."""

from __future__ import annotations

import hashlib
import logging

from app.domain.interfaces import LLM
from app.common.cache import Cache

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        llm: LLM,
        cache: Cache | None = None,
        default_temperature: float = 0.2,
        default_max_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        cacheable: bool = True,
    ) -> str:
        if not prompt.strip():
            raise ValueError("refusing to call the LLM with an empty prompt")

        key = self._cache_key(prompt, system_prompt, temperature, max_tokens)
        if self.cache is not None and cacheable:
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug("LLM response cache hit")
                return cached

        response = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )

        if self.cache is not None and cacheable:
            self.cache.set(key, response)
        return response

    def _cache_key(self, prompt: str, system: str | None, temperature: float | None, max_tokens: int | None) -> str:
        payload = "\x00".join(
            [
                self.llm.model_name,
                system or "",
                prompt,
                f"{temperature if temperature is not None else self.default_temperature}",
                f"{max_tokens if max_tokens is not None else self.default_max_tokens}",
            ]
        )
        return f"llm:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"