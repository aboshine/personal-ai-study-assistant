from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from study_assistant.config import Settings
from study_assistant.llm import complete, get_llm_client
from study_assistant.llm.errors import (
    LLMConnectionError,
    LLMResponseError,
    UnsupportedLLMProviderError,
)
from study_assistant.llm.ollama import OllamaClient


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "test",
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
        "llm_base_url": "http://127.0.0.1:11434",
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        "data_dir": Path("."),
        "vector_store_path": Path("vector_store.sqlite"),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _urlopen_response(payload: dict[str, object] | bytes) -> MagicMock:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_get_llm_client_returns_ollama_from_settings() -> None:
    client = get_llm_client(_settings())
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.2"
    assert client.base_url == "http://127.0.0.1:11434"


def test_get_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedLLMProviderError, match="cloud"):
        get_llm_client(_settings(llm_provider="cloud"))


def test_complete_returns_model_text() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")
    fake = _urlopen_response({"response": "Hello from the model", "done": True})

    with patch("study_assistant.llm.ollama.urllib.request.urlopen", return_value=fake) as mocked:
        text = client.complete("Say hello")

    assert text == "Hello from the model"
    mocked.assert_called_once()
    request = mocked.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    sent = json.loads(request.data.decode("utf-8"))
    assert sent == {"model": "llama3.2", "prompt": "Say hello", "stream": False}


def test_complete_helper_uses_factory_and_http() -> None:
    fake = _urlopen_response({"response": "ok"})

    with patch("study_assistant.llm.ollama.urllib.request.urlopen", return_value=fake):
        text = complete("ping", settings=_settings())

    assert text == "ok"


def test_complete_connection_failure() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")

    with patch(
        "study_assistant.llm.ollama.urllib.request.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(LLMConnectionError, match="Cannot connect to Ollama"):
            client.complete("Hello")


def test_complete_http_error() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")
    error = HTTPError(
        "http://127.0.0.1:11434/api/generate",
        500,
        "Internal Server Error",
        hdrs=None,
        fp=BytesIO(b'{"error":"boom"}'),
    )

    with patch("study_assistant.llm.ollama.urllib.request.urlopen", side_effect=error):
        with pytest.raises(LLMResponseError, match="HTTP 500"):
            client.complete("Hello")


def test_complete_invalid_json() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")

    with patch(
        "study_assistant.llm.ollama.urllib.request.urlopen",
        return_value=_urlopen_response(b"not-json"),
    ):
        with pytest.raises(LLMResponseError, match="non-JSON"):
            client.complete("Hello")


def test_complete_missing_response_field() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")

    with patch(
        "study_assistant.llm.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"done": True}),
    ):
        with pytest.raises(LLMResponseError, match="response"):
            client.complete("Hello")


def test_complete_rejects_empty_prompt() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "llama3.2")
    with pytest.raises(LLMResponseError, match="empty"):
        client.complete("   ")
