"""Deterministic RAG reliability checks. No Ollama or network calls."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from study_assistant.embeddings import EmbeddedChunk, EmbeddingClient
from study_assistant.llm.base import LLMClient
from study_assistant.rag import NO_CONTEXT_ANSWER, RagPipeline, build_rag_prompt
from study_assistant.vector_store import SqliteVectorStore, VectorStoreError


class FakeEmbeddings(EmbeddingClient):
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.embedded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return list(self.vector)


class FakeLLM(LLMClient):
    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.response is not None:
            return self.response
        numbers = _prompt_source_numbers(prompt)
        return " ".join(f"fact [{number}]" for number in numbers)


def _stored(
    chunk_id: str,
    text: str,
    vector: tuple[float, ...],
    *,
    source: str = "notes.pdf",
    page: int = 1,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        source_path=Path(source),
        page_number=page,
        vector=vector,
    )


def _prompt_source_numbers(prompt: str) -> list[int]:
    section = prompt.split("Sources:\n", 1)[1].split("\n\nRetrieved context:", 1)[0]
    return [int(match) for match in re.findall(r"^\[(\d+)\]", section, flags=re.MULTILINE)]


def _answer_citation_numbers(answer: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", answer)]


def test_relevant_chunks_rank_above_less_relevant(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("notes.pdf:p1:c1", "Stacks use LIFO.", (1.0, 0.0, 0.0), page=1),
                _stored("notes.pdf:p2:c1", "Related but weaker.", (0.8, 0.6, 0.0), page=2),
                _stored("notes.pdf:p3:c1", "Unrelated trees.", (0.0, 1.0, 0.0), page=3),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=3).answer("What is a stack?")

    assert [source.chunk_id for source in result.sources] == [
        "notes.pdf:p1:c1",
        "notes.pdf:p2:c1",
        "notes.pdf:p3:c1",
    ]
    assert result.sources[0].text == "Stacks use LIFO."
    assert result.sources[0].similarity > result.sources[1].similarity > result.sources[2].similarity


def test_multi_source_paths_and_pages_are_preserved(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("ds.pdf:p4:c1", "Queue is FIFO.", (0.0, 1.0), source="ds.pdf", page=4),
                _stored("notes.pdf:p1:c1", "Stack is LIFO.", (1.0, 0.0), source="notes.pdf", page=1),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=2).answer("What is a stack?")

    assert result.sources[0].source_path == Path("notes.pdf")
    assert result.sources[0].page_number == 1
    assert result.sources[1].source_path == Path("ds.pdf")
    assert result.sources[1].page_number == 4
    prompt = llm.prompts[0]
    assert "[1] notes.pdf, page 1" in prompt
    assert "[2] ds.pdf, page 4" in prompt


def test_citation_numbers_match_retrieved_source_order(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("lab.pdf:p9:c1", "Wear goggles.", (0.2, 0.98), source="lab.pdf", page=9),
                _stored("notes.pdf:p2:c1", "LIFO stack.", (1.0, 0.0), source="notes.pdf", page=2),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=2).answer("Cite the stack definition.")

    cited = _answer_citation_numbers(result.answer)
    assert cited == [1, 2]
    assert all(1 <= number <= len(result.sources) for number in cited)
    assert result.sources[cited[0] - 1].source_path == Path("notes.pdf")
    assert result.sources[cited[0] - 1].page_number == 2
    assert result.sources[cited[1] - 1].source_path == Path("lab.pdf")
    assert result.sources[cited[1] - 1].page_number == 9
    assert _prompt_source_numbers(llm.prompts[0]) == [1, 2]
    second = build_rag_prompt("Cite the stack definition.", result.sources)
    assert _prompt_source_numbers(second) == [1, 2]


def test_empty_retrieval_does_not_call_llm(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM("unused")
    with SqliteVectorStore(tmp_path / "empty.sqlite") as store:
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm).answer("Anything?")

    assert result.sources == ()
    assert result.answer == NO_CONTEXT_ANSWER
    assert llm.prompts == []


def test_retrieval_persists_across_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.sqlite"
    writer = SqliteVectorStore(db_path)
    writer.upsert([_stored("notes.pdf:p1:c1", "Stacks use LIFO.", (1.0, 0.0), page=1)])
    writer.close()

    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    reader = SqliteVectorStore(db_path)
    result = RagPipeline(embeddings=embeddings, store=reader, llm=llm, top_k=1).answer("What is a stack?")
    reader.close()

    assert [source.chunk_id for source in result.sources] == ["notes.pdf:p1:c1"]
    assert result.sources[0].text == "Stacks use LIFO."
    assert result.sources[0].similarity == pytest.approx(1.0)


def test_top_k_one_returns_only_best_chunk(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("a", "best", (1.0, 0.0)),
                _stored("b", "worse", (0.0, 1.0)),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=1).answer("query")

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "a"
    assert _prompt_source_numbers(llm.prompts[0]) == [1]
    assert "Use only citation numbers from [1] to [1]" in llm.prompts[0]
    assert "[2] " not in llm.prompts[0].split("Sources:\n", 1)[1]


def test_top_k_larger_than_store_returns_all_chunks(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("a", "a", (1.0, 0.0)),
                _stored("b", "b", (0.0, 1.0)),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=9).answer("query")

    assert len(result.sources) == 2
    assert _prompt_source_numbers(llm.prompts[0]) == [1, 2]


def test_upsert_replaces_duplicate_chunk_id_for_retrieval(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert([_stored("notes.pdf:p1:c1", "old", (0.0, 1.0), page=1)])
        store.upsert([_stored("notes.pdf:p1:c1", "new LIFO text", (1.0, 0.0), page=8)])
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=1).answer("query")
        assert store.count() == 1

    assert result.sources[0].text == "new LIFO text"
    assert result.sources[0].page_number == 8
    assert result.sources[0].similarity == pytest.approx(1.0)


def test_whitespace_question_is_rejected(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        pipeline = RagPipeline(embeddings=embeddings, store=store, llm=llm)
        with pytest.raises(ValueError, match="empty"):
            pipeline.answer("\t  \n")
    assert embeddings.embedded == []
    assert llm.prompts == []


def test_zero_query_vector_is_rejected(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([0.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert([_stored("a", "text", (1.0, 0.0))])
        with pytest.raises(VectorStoreError, match="zero magnitude"):
            RagPipeline(embeddings=embeddings, store=store, llm=llm).answer("query")
    assert llm.prompts == []


def test_zero_stored_vector_scores_zero(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert(
            [
                _stored("zero", "empty embedding", (0.0, 0.0)),
                _stored("hit", "aligned", (1.0, 0.0)),
            ]
        )
        result = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=2).answer("query")

    assert result.sources[0].chunk_id == "hit"
    assert result.sources[1].chunk_id == "zero"
    assert result.sources[1].similarity == pytest.approx(0.0)


def test_incompatible_query_dimension_is_rejected(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        store.upsert([_stored("a", "three dims", (1.0, 0.0, 0.0))])
        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            RagPipeline(embeddings=embeddings, store=store, llm=llm).answer("query")
    assert llm.prompts == []


def test_same_query_and_database_are_deterministic(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    with SqliteVectorStore(db_path) as store:
        store.upsert(
            [
                _stored("b", "tie-b", (0.6, 0.8)),
                _stored("a", "tie-a", (0.6, 0.8)),
                _stored("best", "best", (1.0, 0.0)),
            ]
        )
        first = store.search((1.0, 0.0), top_k=3)
        second = store.search((1.0, 0.0), top_k=3)

    assert [(item.chunk_id, item.similarity) for item in first] == [
        (item.chunk_id, item.similarity) for item in second
    ]
    assert [item.chunk_id for item in first] == ["best", "a", "b"]
