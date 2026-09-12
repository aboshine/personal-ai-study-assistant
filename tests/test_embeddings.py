from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from study_assistant.chunking import DocumentChunk
from study_assistant.config import Settings
from study_assistant.embeddings import (
    EmbeddedChunk,
    embed_chunks,
    embed_text,
    get_embedding_client,
)
from study_assistant.embeddings.errors import (
    EmbeddingConnectionError,
    EmbeddingResponseError,
    UnsupportedEmbeddingProviderError,
)
from study_assistant.embeddings.ollama import OllamaEmbeddingClient


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
        "quiz_attempt_store_path": Path("quiz_attempts.sqlite"),
        "study_plan_store_path": Path("study_plans.sqlite"),
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


def _chunk(chunk_id: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source_path=Path("notes.pdf"),
        page_number=page,
    )


def test_get_embedding_client_uses_config_model_and_base_url() -> None:
    client = get_embedding_client(_settings())
    assert isinstance(client, OllamaEmbeddingClient)
    assert client.model == "nomic-embed-text"
    assert client.base_url == "http://127.0.0.1:11434"


def test_get_embedding_client_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedEmbeddingProviderError, match="cloud"):
        get_embedding_client(_settings(embedding_provider="cloud"))


def test_embed_returns_numeric_vector() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    fake = _urlopen_response({"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("study_assistant.embeddings.ollama.urllib.request.urlopen", return_value=fake) as mocked:
        vector = client.embed("Stacks use LIFO.")

    assert vector == [0.1, 0.2, 0.3]
    request = mocked.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/embed"
    sent = json.loads(request.data.decode("utf-8"))
    assert sent == {"model": "nomic-embed-text", "input": ["Stacks use LIFO."]}


def test_embed_text_helper_uses_client() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    fake = _urlopen_response({"embeddings": [[1.0, 2.0]]})

    with patch("study_assistant.embeddings.ollama.urllib.request.urlopen", return_value=fake):
        vector = embed_text("hello", client=client)

    assert vector == [1.0, 2.0]


def test_embed_chunks_preserves_metadata_and_ids() -> None:
    chunks = (
        _chunk("notes.pdf:p1:c1", "first chunk", page=1),
        _chunk("notes.pdf:p2:c1", "second chunk", page=2),
    )
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    fake = _urlopen_response({"embeddings": [[0.1, 0.0], [0.0, 0.2]]})

    with patch("study_assistant.embeddings.ollama.urllib.request.urlopen", return_value=fake) as mocked:
        embedded = embed_chunks(chunks, client=client)

    assert len(embedded) == 2
    assert isinstance(embedded[0], EmbeddedChunk)
    assert embedded[0].chunk_id == "notes.pdf:p1:c1"
    assert embedded[0].text == "first chunk"
    assert embedded[0].source_path == Path("notes.pdf")
    assert embedded[0].page_number == 1
    assert embedded[0].vector == (0.1, 0.0)
    assert embedded[1].chunk_id == "notes.pdf:p2:c1"
    assert embedded[1].page_number == 2
    assert embedded[1].vector == (0.0, 0.2)
    sent = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
    assert sent["input"] == ["first chunk", "second chunk"]


def test_embed_connection_failure() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(EmbeddingConnectionError, match="Cannot connect to Ollama"):
            client.embed("hello")


def test_embed_http_error() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    error = HTTPError(
        "http://127.0.0.1:11434/api/embed",
        404,
        "Not Found",
        hdrs=None,
        fp=BytesIO(b'{"error":"model not found"}'),
    )

    with patch("study_assistant.embeddings.ollama.urllib.request.urlopen", side_effect=error):
        with pytest.raises(EmbeddingResponseError, match="HTTP 404"):
            client.embed("hello")


def test_embed_invalid_json() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response(b"not-json"),
    ):
        with pytest.raises(EmbeddingResponseError, match="non-JSON"):
            client.embed("hello")


def test_embed_missing_embeddings_field() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"model": "nomic-embed-text"}),
    ):
        with pytest.raises(EmbeddingResponseError, match="embeddings"):
            client.embed("hello")


def test_embed_empty_vector() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"embeddings": [[]]}),
    ):
        with pytest.raises(EmbeddingResponseError, match="empty"):
            client.embed("hello")


def test_embed_non_numeric_vector() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"embeddings": [["not-a-number", 1]]}),
    ):
        with pytest.raises(EmbeddingResponseError, match="non-numeric"):
            client.embed("hello")


def test_embed_rejects_empty_text() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    with pytest.raises(EmbeddingResponseError, match="empty"):
        client.embed("   ")
