"""Embeddings public API: config-backed client, text embed, and chunk batch embed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from study_assistant.chunking import DocumentChunk
from study_assistant.config import Settings, load_settings
from study_assistant.embeddings.base import EmbeddingClient
from study_assistant.embeddings.errors import (
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingResponseError,
    UnsupportedEmbeddingProviderError,
)
from study_assistant.embeddings.ollama import OllamaEmbeddingClient

__all__ = [
    "EmbeddedChunk",
    "EmbeddingClient",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingResponseError",
    "OllamaEmbeddingClient",
    "UnsupportedEmbeddingProviderError",
    "embed_chunks",
    "embed_text",
    "get_embedding_client",
]


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk plus its embedding vector. No vector store is used."""

    chunk_id: str
    text: str
    source_path: Path
    page_number: int
    vector: tuple[float, ...]


def get_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    """Return an embedding client for the configured provider."""
    resolved = settings if settings is not None else load_settings()
    provider = resolved.embedding_provider.strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=resolved.llm_base_url,
            model=resolved.embedding_model,
        )
    raise UnsupportedEmbeddingProviderError(
        f"Unsupported embedding provider: {resolved.embedding_provider}"
    )


def embed_text(text: str, *, client: EmbeddingClient | None = None) -> list[float]:
    """Embed a single string using the configured embedding client."""
    embedding_client = client if client is not None else get_embedding_client()
    return embedding_client.embed(text)


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    *,
    client: EmbeddingClient | None = None,
) -> tuple[EmbeddedChunk, ...]:
    """Embed chunks in batch and keep each chunk id, text, path, and page number."""
    embedding_client = client if client is not None else get_embedding_client()
    vectors = embedding_client.embed_texts([chunk.text for chunk in chunks])
    return tuple(
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_path=chunk.source_path,
            page_number=chunk.page_number,
            vector=tuple(vector),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
