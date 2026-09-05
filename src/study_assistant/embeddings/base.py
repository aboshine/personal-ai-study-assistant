"""Replaceable embedding client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingClient(ABC):
    """Minimal text-embedding interface. Swap implementations without changing callers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one text string and return its numeric vector."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts; return one vector per input, in the same order."""
