"""Deterministic echo provider used for local/offline mode and tests.

It does not answer questions; it summarizes the prompt's context so the full
pipeline (retrieval → context → prompt → response → citations) is exercisable
without any network or API key.
"""

from __future__ import annotations

import re

from app.domain.exceptions import LLMError


class EchoLLM:
    """Mimics a provider returning a canned, context-grounded summary.

    For real answers set ``LLM_PROVIDER=ollama`` (or provider of your choice).
    """

    def __init__(self, temperature: float = 0.2, max_tokens: int = 512) -> None:
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return "echo"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not prompt.strip():
            raise LLMError("refusing to generate from an empty prompt")
        text = self._extract_context(prompt)
        answer = (
            "[echo] I do not have real question-answering capabilities. "
            "This is a deterministic pipeline placeholder. Provide an LLM (e.g. "
            "LLM_PROVIDER=ollama) for grounded answers. Evidence available in "
            f"{len(text):d} context block(s)."
        )
        requested = (max_tokens or self._max_tokens)
        return answer[: requested]

    @staticmethod
    def _extract_context(prompt: str) -> list[dict]:
        blocks = re.findall(r"\[Source (\d+)\]\nDocument: (.+?)\nPage: (.+?)\n\n(.*?)(?=\n\n\[Source|\n\nUSER QUESTION|$)", prompt, re.S)
        return [{"n": n, "doc": d, "page": p, "text": t.strip()} for n, d, p, t in blocks]