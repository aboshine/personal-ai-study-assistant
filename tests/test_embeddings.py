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
    embed,
    embed_chunks,
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


def _chunk(text: str, *, chunk_id: str = "notes.pdf:p1:c1", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source_path=Path("notes.pdf"),
        page_number=page,
    )


def test_get_embedding_client_returns_ollama_from_settings() -> None:
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

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=fake,
    ) as mocked:
        vector = client.embed("What is a stack?")

    assert vector == [0.1, 0.2, 0.3]
    mocked.assert_called_once()
    request = mocked.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/embed"
    sent = json.loads(request.data.decode("utf-8"))
    assert sent == {"model": "nomic-embed-text", "input": "What is a stack?"}


def test_embed_helper_uses_factory_and_http() -> None:
    fake = _urlopen_response({"embeddings": [[1.0, 2.0]]})

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=fake,
    ):
        vector = embed("ping", settings=_settings())

    assert vector == [1.0, 2.0]


def test_embed_batch_preserves_order() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    fake = _urlopen_response({"embeddings": [[1.0], [2.0], [3.0]]})

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=fake,
    ) as mocked:
        vectors = client.embed_batch(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    sent = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
    assert sent["input"] == ["a", "b", "c"]


def test_embed_batch_empty_input_returns_empty_list() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    assert client.embed_batch([]) == []


def test_embed_chunks_preserves_metadata_and_ids() -> None:
    chunks = (
        _chunk("first chunk", chunk_id="notes.pdf:p1:c1", page=1),
        _chunk("second chunk", chunk_id="notes.pdf:p2:c1", page=2),
    )
    fake = _urlopen_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=fake,
    ):
        embedded = embed_chunks(chunks, settings=_settings())

    assert len(embedded) == 2
    assert embedded[0] == EmbeddedChunk(
        chunk_id="notes.pdf:p1:c1",
        text="first chunk",
        source_path=Path("notes.pdf"),
        page_number=1,
        embedding=(0.1, 0.2),
    )
    assert embedded[1].chunk_id == "notes.pdf:p2:c1"
    assert embedded[1].page_number == 2
    assert embedded[1].embedding == (0.3, 0.4)


def test_embed_chunks_empty_returns_empty_tuple() -> None:
    assert embed_chunks([], settings=_settings()) == ()


def test_embed_connection_failure() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(EmbeddingConnectionError, match="Cannot connect to Ollama"):
            client.embed("Hello")


def test_embed_http_error() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    error = HTTPError(
        "http://127.0.0.1:11434/api/embed",
        500,
        "Internal Server Error",
        hdrs=None,
        fp=BytesIO(b'{"error":"boom"}'),
    )

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        side_effect=error,
    ):
        with pytest.raises(EmbeddingResponseError, match="HTTP 500"):
            client.embed("Hello")


def test_embed_invalid_json() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response(b"not-json"),
    ):
        with pytest.raises(EmbeddingResponseError, match="non-JSON"):
            client.embed("Hello")


def test_embed_missing_embeddings_field() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"model": "nomic-embed-text"}),
    ):
        with pytest.raises(EmbeddingResponseError, match="embeddings"):
            client.embed("Hello")


def test_embed_empty_embeddings_list() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"embeddings": []}),
    ):
        with pytest.raises(EmbeddingResponseError, match="empty"):
            client.embed("Hello")


def test_embed_empty_vector() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"embeddings": [[]]}),
    ):
        with pytest.raises(EmbeddingResponseError, match="empty vector"):
            client.embed("Hello")


def test_embed_non_numeric_vector_values() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=_urlopen_response({"embeddings": [["bad", 1.0]]}),
    ):
        with pytest.raises(EmbeddingResponseError, match="non-numeric"):
            client.embed("Hello")


def test_embed_rejects_empty_text() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    with pytest.raises(EmbeddingResponseError, match="empty"):
        client.embed("   ")


def test_embed_batch_rejects_empty_item() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    with pytest.raises(EmbeddingResponseError, match="index 1"):
        client.embed_batch(["ok", "  "])


def test_embed_batch_count_mismatch() -> None:
    client = OllamaEmbeddingClient("http://127.0.0.1:11434", "nomic-embed-text")
    fake = _urlopen_response({"embeddings": [[1.0]]})

    with patch(
        "study_assistant.embeddings.ollama.urllib.request.urlopen",
        return_value=fake,
    ):
        with pytest.raises(EmbeddingResponseError, match="returned 1 embeddings for 2"):
            client.embed_batch(["a", "b"])
