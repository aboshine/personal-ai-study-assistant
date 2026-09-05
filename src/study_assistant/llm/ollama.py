"""Ollama client using the local HTTP generate API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from study_assistant.llm.base import LLMClient
from study_assistant.llm.errors import LLMConnectionError, LLMResponseError

_GENERATE_PATH = "/api/generate"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaClient(LLMClient):
    """Calls Ollama at `{base_url}/api/generate` with `stream` disabled."""

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
            raise LLMResponseError("Ollama base URL is empty")
        if not cleaned_model:
            raise LLMResponseError("Ollama model name is empty")
        self.base_url = cleaned_url
        self.model = cleaned_model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        if not prompt.strip():
            raise LLMResponseError("Prompt must not be empty")

        request = urllib.request.Request(
            f"{self.base_url}{_GENERATE_PATH}",
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
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
            raise LLMResponseError(
                f"Ollama HTTP {exc.code} at {self.base_url}: {exc.reason}{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from exc
        except TimeoutError as exc:
            raise LLMConnectionError(
                f"Timed out waiting for Ollama at {self.base_url}"
            ) from exc

        return _parse_generate_response(raw_body)


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not body:
        return ""
    return f" ({body})"


def _parse_generate_response(raw_body: bytes) -> str:
    try:
        payload: Any = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMResponseError("Ollama returned a non-JSON response") from exc

    if not isinstance(payload, dict):
        raise LLMResponseError("Ollama JSON response must be an object")

    text = payload.get("response")
    if not isinstance(text, str):
        raise LLMResponseError("Ollama JSON response is missing a string 'response' field")
    return text
