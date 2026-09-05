"""Errors raised by the LLM layer."""


class LLMError(Exception):
    """Base error for LLM client failures."""


class LLMConnectionError(LLMError):
    """Raised when the provider cannot be reached."""


class LLMResponseError(LLMError):
    """Raised when the provider returns an invalid or unsuccessful response."""


class UnsupportedLLMProviderError(LLMError):
    """Raised when config names a provider that is not implemented."""
