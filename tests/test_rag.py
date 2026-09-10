from pathlib import Path

import pytest

from study_assistant.embeddings import EmbeddedChunk, EmbeddingClient
from study_assistant.llm.base import LLMClient
from study_assistant.rag import NO_CONTEXT_ANSWER, RagPipeline, build_rag_prompt
from study_assistant.vector_store import SimilarChunk, SqliteVectorStore


class FakeEmbeddings(EmbeddingClient):
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.embedded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return list(self.vector)


class FakeLLM(LLMClient):
    def __init__(self, response: str = "Grounded answer.") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _similar(
    chunk_id: str,
    text: str,
    *,
    source: str = "notes.pdf",
    page: int = 1,
    similarity: float = 0.9,
) -> SimilarChunk:
    return SimilarChunk(
        chunk_id=chunk_id,
        source_path=Path(source),
        page_number=page,
        text=text,
        similarity=similarity,
    )


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


def test_build_rag_prompt_includes_context_rules_and_metadata() -> None:
    chunks = (
        _similar("notes.pdf:p1:c1", "Stacks use LIFO.", page=1, similarity=0.91),
        _similar("notes.pdf:p2:c1", "Queues use FIFO.", page=2, similarity=0.4),
    )

    prompt = build_rag_prompt("What is a stack?", chunks)

    assert "ONLY the retrieved context" in prompt
    assert "Do not invent facts" in prompt
    assert "Cite factual claims with [1], [2]" in prompt
    assert "Never invent citations" in prompt
    assert "Use only citation numbers from [1] to [2]" in prompt
    assert prompt.index("[1] notes.pdf, page 1") < prompt.index("[2] notes.pdf, page 2")
    assert prompt.index("[1] source=notes.pdf page=1") < prompt.index("[2] source=notes.pdf page=2")
    assert "chunk_id=notes.pdf:p1:c1" in prompt
    assert "Stacks use LIFO." in prompt
    assert "Queues use FIFO." in prompt
    assert "What is a stack?" in prompt


def test_build_rag_prompt_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="without retrieved chunks"):
        build_rag_prompt("Anything?", ())


def test_rag_pipeline_embeds_retrieves_and_answers(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    embeddings = FakeEmbeddings([1.0, 0.0, 0.0])
    llm = FakeLLM("A stack uses LIFO. [1]")

    with SqliteVectorStore(db_path) as store:
        store.upsert(
            [
                _stored("notes.pdf:p1:c1", "Stacks use LIFO.", (1.0, 0.0, 0.0), page=1),
                _stored("notes.pdf:p2:c1", "Trees have nodes.", (0.6, 0.8, 0.0), page=2),
                _stored("lab.pdf:p1:c1", "Lab safety first.", (0.0, 1.0, 0.0), source="lab.pdf"),
            ]
        )
        pipeline = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=2)
        result = pipeline.answer("What is a stack?")

    assert embeddings.embedded == ["What is a stack?"]
    assert result.answer == "A stack uses LIFO. [1]"
    assert [source.chunk_id for source in result.sources] == ["notes.pdf:p1:c1", "notes.pdf:p2:c1"]
    assert result.sources[0].page_number == 1
    assert result.sources[0].source_path == Path("notes.pdf")
    assert result.sources[0].similarity == pytest.approx(1.0)
    assert result.sources[1].similarity == pytest.approx(0.6)
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert prompt.index("[1] notes.pdf, page 1") < prompt.index("[2] notes.pdf, page 2")
    assert "Cite factual claims with [1], [2]" in prompt
    assert "Use only citation numbers from [1] to [2]" in prompt
    assert "Stacks use LIFO." in prompt
    assert "Trees have nodes." in prompt
    assert "Lab safety first." not in prompt
    assert "source=notes.pdf page=1" in prompt
    assert "What is a stack?" in prompt


def test_rag_pipeline_empty_retrieval_skips_llm(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.sqlite"
    embeddings = FakeEmbeddings([1.0, 0.0])
    llm = FakeLLM("should not be used")

    with SqliteVectorStore(db_path) as store:
        pipeline = RagPipeline(embeddings=embeddings, store=store, llm=llm, top_k=3)
        result = pipeline.answer("What is a stack?")

    assert embeddings.embedded == ["What is a stack?"]
    assert result.sources == ()
    assert result.answer == NO_CONTEXT_ANSWER
    assert llm.prompts == []


def test_rag_pipeline_rejects_empty_question(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings([1.0])
    llm = FakeLLM()
    with SqliteVectorStore(tmp_path / "store.sqlite") as store:
        pipeline = RagPipeline(embeddings=embeddings, store=store, llm=llm)
        with pytest.raises(ValueError, match="empty"):
            pipeline.answer("   ")
    assert embeddings.embedded == []
    assert llm.prompts == []
