import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from study_assistant.embeddings import EmbeddedChunk, EmbeddingClient
from study_assistant.llm.base import LLMClient
from study_assistant.pdf_extraction import InvalidPdfError, PdfNotFoundError
from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore
from study_assistant.quiz_evaluation import QuizEvaluationError, SubmittedAnswer
from study_assistant.rag import RagPipeline
from study_assistant.service import StudyAssistant
from study_assistant.topic_tracking import summarize_topic_performance
from study_assistant.vector_store import SqliteVectorStore
from tests.pdf_fixtures import minimal_pdf_bytes

AS_OF = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class FakeEmbeddings(EmbeddingClient):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "multiple-choice quizzes" in prompt:
            if "Generate exactly 1 question(s)." in prompt:
                return _quiz_json(1, topic="Queues")
            return _quiz_json(2)
        return "A stack uses LIFO. [1]"


def _quiz_json(count: int, *, topic: str = "Stacks") -> str:
    questions = []
    topics = ("Stacks", "Queues") if count == 2 else (topic,)
    for index, name in enumerate(topics):
        questions.append(
            {
                "question": f"What is true of {name}?",
                "options": [
                    {"label": "A", "text": f"{name}-A-{index}"},
                    {"label": "B", "text": f"{name}-B-{index}"},
                    {"label": "C", "text": f"{name}-C-{index}"},
                    {"label": "D", "text": f"{name}-D-{index}"},
                ],
                "correct_label": "A" if name == "Stacks" else "B",
                "explanation": f"{name} fact.",
                "topic": name,
                "source_citations": [1],
            }
        )
    return json.dumps({"questions": questions})


def _chunk(chunk_id: str, text: str, vector: tuple[float, ...], *, page: int) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        source_path=Path("notes.pdf"),
        page_number=page,
        vector=vector,
    )


def _assistant(tmp_path: Path) -> tuple[StudyAssistant, FakeLLM, SqliteVectorStore, SqliteQuizAttemptStore]:
    llm = FakeLLM()
    vector_store = SqliteVectorStore(tmp_path / "vectors.sqlite")
    vector_store.upsert(
        [
            _chunk("notes.pdf:p1:c1", "Stacks use LIFO ordering.", (1.0, 0.0, 0.0), page=1),
            _chunk("notes.pdf:p2:c1", "Queues use FIFO ordering.", (0.9, 0.1, 0.0), page=2),
        ]
    )
    rag = RagPipeline(embeddings=FakeEmbeddings(), store=vector_store, llm=llm, top_k=2)
    attempts = SqliteQuizAttemptStore(tmp_path / "attempts.sqlite")
    assistant = StudyAssistant(rag=rag, attempt_store=attempts)
    return assistant, llm, vector_store, attempts


def _empty_assistant(tmp_path: Path) -> tuple[StudyAssistant, SqliteVectorStore, SqliteQuizAttemptStore]:
    vector_store = SqliteVectorStore(tmp_path / "vectors.sqlite")
    attempts = SqliteQuizAttemptStore(tmp_path / "attempts.sqlite")
    assistant = StudyAssistant(
        rag=RagPipeline(embeddings=FakeEmbeddings(), store=vector_store, llm=FakeLLM(), top_k=2),
        attempt_store=attempts,
    )
    return assistant, vector_store, attempts


def test_index_pdf_stores_extracted_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(minimal_pdf_bytes(["A stack uses LIFO ordering.", "A queue uses FIFO ordering."]))
    assistant, store, attempts = _empty_assistant(tmp_path)

    result = assistant.index_pdf(pdf_path)

    assert result.source_path == pdf_path.resolve()
    assert result.chunk_count >= 1
    assert store.count() == result.chunk_count
    stored = store.list_by_source(pdf_path.resolve())
    assert len(stored) == result.chunk_count
    assert any("LIFO" in chunk.text for chunk in stored)
    store.close()
    attempts.close()


def test_reindex_replaces_old_chunks_for_the_same_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(minimal_pdf_bytes(["A" * 2000, "B" * 2000]))
    assistant, store, attempts = _empty_assistant(tmp_path)
    first = assistant.index_pdf(pdf_path)
    other = EmbeddedChunk(
        chunk_id="other.pdf:p1:c1",
        text="Unrelated notes.",
        source_path=Path("other.pdf"),
        page_number=1,
        vector=(0.0, 1.0, 0.0),
    )
    store.upsert([other])

    pdf_path.write_bytes(minimal_pdf_bytes(["Short LIFO note."]))
    second = assistant.index_pdf(pdf_path)

    assert first.chunk_count > 1
    assert second.chunk_count < first.chunk_count
    assert store.count() == second.chunk_count + 1
    assert len(store.list_by_source(pdf_path.resolve())) == second.chunk_count
    assert store.get("other.pdf:p1:c1") is not None
    store.close()
    attempts.close()


def test_index_pdf_rejects_missing_and_invalid_files(tmp_path: Path) -> None:
    assistant, store, attempts = _empty_assistant(tmp_path)

    with pytest.raises(PdfNotFoundError):
        assistant.index_pdf(tmp_path / "missing.pdf")

    invalid = tmp_path / "notes.pdf"
    invalid.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(InvalidPdfError):
        assistant.index_pdf(invalid)

    assert store.count() == 0
    store.close()
    attempts.close()


def test_ask_uses_rag_over_retrieved_chunks(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempts = _assistant(tmp_path)
    result = assistant.ask("What is a stack?")
    assert "LIFO" in result.answer
    assert result.sources
    assert any("Stacks use LIFO" in chunk.text for chunk in result.sources)
    assert "ONLY the retrieved context" in llm.prompts[0]
    vector_store.close()
    attempts.close()


def test_evaluate_and_record_persists_attempt_for_adaptive_and_plan(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempts = _assistant(tmp_path)
    quiz = assistant.generate_quiz("data structures", question_count=2)
    evaluation, recorded = assistant.evaluate_and_record(
        quiz,
        (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")),
        timestamp=AS_OF,
    )
    assert evaluation.correct_count == 1
    assert attempts.get(recorded.attempt_id) == recorded
    assert attempts.list_all() == (recorded,)
    assert assistant.list_attempts() == (recorded,)

    adaptive = assistant.generate_adaptive_quiz(
        "data structures",
        question_count=1,
        available_topics=("Stacks", "Queues"),
    )
    assert "Prioritize these topics (weakest first): Queues" in llm.prompts[-1]
    assert adaptive.difficulty == "easy"

    plan = assistant.current_study_plan(as_of=AS_OF)
    assert plan.items[0].topic == "Queues"

    vector_store.close()
    attempts.close()


def test_evaluate_and_record_does_not_persist_invalid_answers(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempts = _assistant(tmp_path)
    quiz = assistant.generate_quiz("data structures", question_count=2)
    with pytest.raises(QuizEvaluationError):
        assistant.evaluate_and_record(quiz, (SubmittedAnswer(0, "E"),))
    assert attempts.list_all() == ()
    vector_store.close()
    attempts.close()


def test_quiz_evaluate_record_feeds_topic_tracking_and_plan(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempts = _assistant(tmp_path)
    quiz = assistant.generate_quiz("data structures", question_count=2, difficulty="medium")
    assert len(quiz.questions) == 2
    assert {question.topic for question in quiz.questions} == {"Stacks", "Queues"}
    assert quiz.questions[0].sources[0].chunk_id

    evaluation = assistant.evaluate_quiz(
        quiz,
        (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")),
    )
    assert evaluation.correct_count == 1
    assert evaluation.incorrect_count == 1

    recorded = assistant.record_attempt(quiz, evaluation, timestamp=AS_OF)
    assert attempts.get(recorded.attempt_id) == recorded

    stats = summarize_topic_performance(attempts.list_all())
    by_topic = {item.topic: item for item in stats}
    assert by_topic["Stacks"].accuracy == 100.0
    assert by_topic["Queues"].accuracy == 0.0

    plan = assistant.current_study_plan(as_of=AS_OF)
    assert plan.items[0].topic == "Queues"
    assert plan.items[0].recommended_difficulty == "easy"
    assert plan.items[1].topic == "Stacks"
    assert plan.items[1].recommended_difficulty == "hard"

    vector_store.close()
    attempts.close()


def test_adaptive_quiz_uses_updated_history(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempts = _assistant(tmp_path)
    quiz = assistant.generate_quiz("data structures", question_count=2)
    evaluation = assistant.evaluate_quiz(
        quiz,
        (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")),
    )
    assistant.record_attempt(quiz, evaluation, timestamp=AS_OF)

    adaptive = assistant.generate_adaptive_quiz(
        "data structures",
        question_count=1,
        available_topics=("Stacks", "Queues"),
    )
    adaptive_prompt = llm.prompts[-1]
    assert "Prioritize these topics (weakest first): Queues" in adaptive_prompt
    assert "Difficulty: easy." in adaptive_prompt
    assert adaptive.difficulty == "easy"
    assert len(adaptive.questions) == 1

    vector_store.close()
    attempts.close()
