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
        if not text.strip():
            raise EmbeddingResponseError("Text to embed must not be empty")
        vectors = self._request_embeddings(text)
        if len(vectors) != 1:
            raise EmbeddingResponseError(
                f"Ollama returned {len(vectors)} embeddings for a single input; expected 1"
            )
        return vectors[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingResponseError(
                    f"Batch item at index {index} must be a non-empty string"
                )
            cleaned.append(text)
        vectors = self._request_embeddings(cleaned)
        if len(vectors) != len(cleaned):
            raise EmbeddingResponseError(
                f"Ollama returned {len(vectors)} embeddings for {len(cleaned)} inputs"
            )
        return vectors

    def _request_embeddings(self, input_value: str | list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}{_EMBED_PATH}",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": input_value,
                }
            ).encode("utf-8"),
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

        return _parse_embed_response(raw_body)


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not body:
        return ""
    return f" ({body})"


def _parse_embed_response(raw_body: bytes) -> list[list[float]]:
    try:
        payload: Any = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingResponseError("Ollama returned a non-JSON embedding response") from exc

    if not isinstance(payload, dict):
        raise EmbeddingResponseError("Ollama embedding JSON response must be an object")

    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise EmbeddingResponseError(
            "Ollama embedding JSON response is missing an 'embeddings' field"
        )
    if not isinstance(embeddings, list):
        raise EmbeddingResponseError(
            "Ollama embedding JSON 'embeddings' field must be a list"
        )
    if len(embeddings) == 0:
        raise EmbeddingResponseError("Ollama embedding JSON 'embeddings' list is empty")

    vectors: list[list[float]] = []
    for index, item in enumerate(embeddings):
        vectors.append(_validate_vector(item, index))
    return vectors


def _validate_vector(item: Any, index: int) -> list[float]:
    if not isinstance(item, list):
        raise EmbeddingResponseError(
            f"Ollama embedding at index {index} must be a list of numbers"
        )
    if len(item) == 0:
        raise EmbeddingResponseError(f"Ollama embedding at index {index} is an empty vector")
    vector: list[float] = []
    for dim_index, value in enumerate(item):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingResponseError(
                f"Ollama embedding at index {index} has a non-numeric value at dimension {dim_index}"
            )
        vector.append(float(value))
    return vector
