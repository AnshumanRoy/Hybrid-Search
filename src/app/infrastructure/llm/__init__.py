"""LLM infrastructure: local providers implemented behind the ``LLM`` protocol.

Ordinary providers (OpenAI, Azure, Gemini, Anthropic, Bedrock, ...) can be
added by implementing ``generate()`` and registering the class here.
"""

from app.infrastructure.llm.echo import EchoLLM
from app.infrastructure.llm.ollama import OllamaLLM

__all__ = ["EchoLLM", "OllamaLLM"]