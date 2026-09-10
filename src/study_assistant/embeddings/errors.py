"""Errors raised by the embeddings layer."""


class EmbeddingError(Exception):
    """Base error for embedding client failures."""


class EmbeddingConnectionError(EmbeddingError):
    """Raised when the embedding provider cannot be reached."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when the provider returns an invalid or unsuccessful response."""


class UnsupportedEmbeddingProviderError(EmbeddingError):
    """Raised when config names an embedding provider that is not implemented."""
