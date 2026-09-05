"""Embeddings public API: config-backed client factory and chunk helpers."""

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
    "embed",
    "embed_chunks",
    "get_embedding_client",
]


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk plus its embedding vector. Metadata mirrors DocumentChunk."""

    chunk_id: str
    text: str
    source_path: Path
    page_number: int
    embedding: tuple[float, ...]


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


def embed(text: str, *, settings: Settings | None = None) -> list[float]:
    """Embed one text string with the configured provider and return its vector."""
    return get_embedding_client(settings).embed(text)


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    *,
    settings: Settings | None = None,
) -> tuple[EmbeddedChunk, ...]:
    """Embed each chunk's text and preserve chunk_id, text, source_path, and page_number."""
    if not chunks:
        return ()

    vectors = get_embedding_client(settings).embed_batch([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingResponseError(
            f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    return tuple(
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_path=chunk.source_path,
            page_number=chunk.page_number,
            embedding=tuple(vector),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
