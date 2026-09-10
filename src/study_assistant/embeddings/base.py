"""Replaceable embedding client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingClient(ABC):
    """Minimal embedding interface. Swap implementations without changing callers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one string and return a numeric vector."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many strings, preserving input order."""
        return [self.embed(text) for text in texts]
