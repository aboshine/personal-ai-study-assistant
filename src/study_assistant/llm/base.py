"""Replaceable LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Minimal text-completion interface. Swap implementations without changing callers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt and return the model text."""
