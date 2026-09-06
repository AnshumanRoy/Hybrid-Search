"""Ollama local LLM provider (OpenAI-compatible /chat/completions endpoint)."""

from __future__ import annotations

import logging

from app.domain.exceptions import LLMError
from app.domain.interfaces import LLM

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    @property
    def model_name(self) -> str:
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
            raise LLMError("httpx is not installed; cannot call Ollama") from exc

        payload = {
            "model": self.model,
            "messages": [],
            "temperature": temperature if temperature is not None else self._temperature,
            "options": {"num_predict": max_tokens or self._max_tokens},
            "stream": False,
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        try:
            with httpx.Client(timeout=timeout_seconds or self._timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise LLMError(f"Ollama request failed (model={self.model}): {exc}") from exc

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMError(f"Ollama returned an empty response (model={self.model})")
        return content