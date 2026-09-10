"""Ollama client using the local HTTP embed API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from study_assistant.embeddings.base import EmbeddingClient
from study_assistant.embeddings.errors import EmbeddingConnectionError, EmbeddingResponseError

_EMBED_PATH = "/api/embed"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaEmbeddingClient(EmbeddingClient):
    """Calls Ollama at `{base_url}/api/embed`."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        cleaned_url = base_url.strip().rstrip("/")
        cleaned_model = model.strip()
        if not cleaned_url:
            raise EmbeddingResponseError("Ollama base URL is empty")
        if not cleaned_model:
            raise EmbeddingResponseError("Ollama embedding model name is empty")
        self.base_url = cleaned_url
        self.model = cleaned_model
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        for text in texts:
            if not text.strip():
                raise EmbeddingResponseError("Text to embed must not be empty")

        request = urllib.request.Request(
            f"{self.base_url}{_EMBED_PATH}",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_body(exc)
            raise EmbeddingResponseError(
                f"Ollama HTTP {exc.code} at {self.base_url}: {exc.reason}{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from exc
        except TimeoutError as exc:
            raise EmbeddingConnectionError(
                f"Timed out waiting for Ollama at {self.base_url}"
            ) from exc

        return _parse_embed_response(raw_body, expected_count=len(texts))


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not body:
        return ""
    return f" ({body})"


def _parse_embed_response(raw_body: bytes, *, expected_count: int) -> list[list[float]]:
    try:
        payload: Any = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingResponseError("Ollama returned a non-JSON embedding response") from exc

    if not isinstance(payload, dict):
        raise EmbeddingResponseError("Ollama embedding JSON response must be an object")

    raw_vectors = payload.get("embeddings")
    if raw_vectors is None and isinstance(payload.get("embedding"), list):
        raw_vectors = [payload["embedding"]]
    if not isinstance(raw_vectors, list) or not raw_vectors:
        raise EmbeddingResponseError("Ollama JSON response is missing a non-empty 'embeddings' list")
    if len(raw_vectors) != expected_count:
        raise EmbeddingResponseError(
            f"Ollama returned {len(raw_vectors)} embeddings but {expected_count} were requested"
        )

    return [_as_vector(item) for item in raw_vectors]


def _as_vector(values: Any) -> list[float]:
    if not isinstance(values, list) or not values:
        raise EmbeddingResponseError("Ollama embedding vector is empty")
    vector: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EmbeddingResponseError("Ollama embedding contains a non-numeric value")
        vector.append(float(item))
    return vector
