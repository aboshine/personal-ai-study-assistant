"""LLM public API: config-backed client factory and a one-shot complete() helper."""

from __future__ import annotations

from study_assistant.config import Settings, load_settings
from study_assistant.llm.base import LLMClient
from study_assistant.llm.errors import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    UnsupportedLLMProviderError,
)
from study_assistant.llm.ollama import OllamaClient

__all__ = [
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMResponseError",
    "OllamaClient",
    "UnsupportedLLMProviderError",
    "complete",
    "get_llm_client",
]


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return an LLM client for the configured provider."""
    resolved = settings if settings is not None else load_settings()
    provider = resolved.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(base_url=resolved.llm_base_url, model=resolved.llm_model)
    raise UnsupportedLLMProviderError(f"Unsupported LLM provider: {resolved.llm_provider}")


def complete(prompt: str, *, settings: Settings | None = None) -> str:
    """Send a text prompt to the configured LLM and return the model response."""
    return get_llm_client(settings).complete(prompt)
