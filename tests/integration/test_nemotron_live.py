"""Optional live smoke tests against the real NVIDIA NIM endpoint.

Skipped unless a real API key + the backend are configured:

    $env:RAG_TEST_NIM_KEY = "nvapi-..."            # or set NVIDIA_API_KEY
    $env:RAG_TEST_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
    python -m pytest tests/integration/test_nemotron_live.py -q

These tests make real network calls and cost tokens, so they are always
skipped in CI / the default run.
"""

from __future__ import annotations

import os

import pytest

from app.infrastructure.llm.nemotron import NemotronNIMClient

KEY = os.environ.get("RAG_TEST_NIM_KEY") or os.environ.get("NVIDIA_API_KEY")
BASE_URL = os.environ.get("RAG_TEST_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

pytestmark = pytest.mark.skipif(
    not KEY,
    reason="set RAG_TEST_NIM_KEY (or NVIDIA_API_KEY) + RAG_TEST_NIM_BASE_URL to run",
)


def test_nim_health_smoke() -> None:
    client = NemotronNIMClient(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key=KEY,
        base_url=BASE_URL,
        max_tokens=16,
        max_retries=1,
        timeout_seconds=30.0,
    )
    answer = client.generate("Reply with the single word: hello")
    assert isinstance(answer, str)
    assert answer.strip()


def test_nim_bad_key_is_reported_as_auth_error() -> None:
    client = NemotronNIMClient(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key="nvapi-definitely-invalid-key-0000000000",
        base_url=BASE_URL,
        max_tokens=8,
        max_retries=1,
        timeout_seconds=30.0,
    )
    from app.domain.exceptions import LLMAuthenticationError

    with pytest.raises(LLMAuthenticationError):
        client.generate("hello")