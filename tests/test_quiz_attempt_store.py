import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from study_assistant.config import Settings
from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_attempt_store import (
    QuizAttemptStoreError,
    SqliteQuizAttemptStore,
    deserialize_question_results,
    get_quiz_attempt_store,
    serialize_question_results,
)
from study_assistant.quiz_attempts import QuizAttempt, create_quiz_attempt
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz


def _options(*texts: str) -> tuple[AnswerOption, ...]:
    return tuple(AnswerOption(label=label, text=text) for label, text in zip("ABCD", texts, strict=True))


def _source(chunk_id: str, *, source: str = "notes.pdf", page: int = 1, citation: int = 1) -> QuestionSource:
    return QuestionSource(
        citation_index=citation,
        chunk_id=chunk_id,
        source_path=Path(source),
        page_number=page,
    )


def _question(
    text: str,
    correct_label: str,
    explanation: str,
    sources: tuple[QuestionSource, ...],
) -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=_options("LIFO", "FIFO", "Heap", "Graph"),
        correct_label=correct_label,
        explanation=explanation,
        sources=sources,
    )


def _quiz(*, quiz_id: str = "quiz-1", difficulty: str = "medium") -> Quiz:
    return Quiz(
        questions=(
            _question(
                "What ordering does a stack use?",
                "A",
                "Stacks use LIFO.",
                (_source("notes.pdf:p1:c1", page=1),),
            ),
            _question(
                "What ordering does a queue use?",
                "B",
                "Queues use FIFO.",
                (_source("ds.pdf:p3:c1", source="ds.pdf", page=3, citation=2),),
            ),
        ),
        difficulty=difficulty,  # type: ignore[arg-type]
        quiz_id=quiz_id,
    )


def _attempt(
    *,
    quiz_id: str = "quiz-1",
    attempt_id: str | None = None,
    timestamp: datetime | None = None,
    labels: tuple[str | None, ...] = ("A", "B"),
    difficulty: str = "medium",
) -> QuizAttempt:
    quiz = _quiz(quiz_id=quiz_id, difficulty=difficulty)
    evaluation = evaluate_quiz(
        quiz,
        tuple(SubmittedAnswer(index, label) for index, label in enumerate(labels)),
    )
    attempt = create_quiz_attempt(
        quiz,
        evaluation,
        timestamp=timestamp or datetime(2026, 9, 10, 2, 30, tzinfo=timezone.utc),
    )
    if attempt_id is None:
        return attempt
    return QuizAttempt(
        quiz_id=attempt.quiz_id,
        timestamp=attempt.timestamp,
        evaluation=attempt.evaluation,
        difficulty=attempt.difficulty,
        question_results=attempt.question_results,
        attempt_id=attempt_id,
    )


def _settings(db_path: Path) -> Settings:
    return Settings(
        app_env="test",
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_base_url="http://127.0.0.1:11434",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        data_dir=db_path.parent,
        vector_store_path=db_path.parent / "vector_store.sqlite",
        quiz_attempt_store_path=db_path,
        study_plan_store_path=db_path.parent / "study_plans.sqlite",
    )


def test_serialize_question_results_is_deterministic() -> None:
    attempt = _attempt()
    encoded = serialize_question_results(attempt.question_results)
    assert encoded == serialize_question_results(attempt.question_results)
    assert deserialize_question_results(encoded) == attempt.question_results
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_empty_store(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    with SqliteQuizAttemptStore(db_path) as store:
        assert store.count() == 0
        assert store.list_all() == ()
        assert store.list_for_quiz("quiz-1") == ()
        assert store.get("missing") is None
        assert store.delete("missing") is False


def test_add_get_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    attempt = _attempt(
        attempt_id="attempt-1",
        labels=("A", None),
        difficulty="hard",
        timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
    )

    with SqliteQuizAttemptStore(db_path) as store:
        stored_id = store.add(attempt)
        assert stored_id == "attempt-1"
        loaded = store.get("attempt-1")

    assert loaded == attempt
    assert loaded is not None
    assert loaded.attempt_id == "attempt-1"
    assert loaded.quiz_id == "quiz-1"
    assert loaded.timestamp == datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    assert loaded.difficulty == "hard"
    assert loaded.evaluation.total_questions == 2
    assert loaded.evaluation.correct_count == 1
    assert loaded.evaluation.incorrect_count == 0
    assert loaded.evaluation.unanswered_count == 1
    assert loaded.evaluation.score_percent == 50.0
    assert loaded.question_results[0].sources[0].source_path == Path("notes.pdf")
    assert loaded.question_results[1].selected_label is None
    assert loaded.question_results[1].is_unanswered is True


def test_persists_across_separate_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "attempts.sqlite"
    attempt = _attempt(attempt_id="persist-1", labels=("C", "B"))

    writer = SqliteQuizAttemptStore(db_path)
    writer.add(attempt)
    writer.close()

    reader = SqliteQuizAttemptStore(db_path)
    loaded = reader.get("persist-1")
    assert loaded == attempt
    assert reader.count() == 1
    reader.close()


def test_get_quiz_attempt_store_uses_settings_path(tmp_path: Path) -> None:
    db_path = tmp_path / "from-settings.sqlite"
    attempt = _attempt(attempt_id="via-settings")
    store = get_quiz_attempt_store(_settings(db_path))
    store.add(attempt)
    store.close()

    reopened = SqliteQuizAttemptStore(db_path)
    assert reopened.get("via-settings") == attempt
    reopened.close()


def test_duplicate_attempt_id_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    first = _attempt(attempt_id="dup", quiz_id="quiz-1")
    second = _attempt(attempt_id="dup", quiz_id="quiz-2", labels=("C", "A"))
    with SqliteQuizAttemptStore(db_path) as store:
        store.add(first)
        with pytest.raises(QuizAttemptStoreError, match="already exists"):
            store.add(second)
        assert store.count() == 1
        assert store.get("dup") == first


def test_list_for_quiz_and_list_all(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    early = _attempt(
        attempt_id="a2",
        quiz_id="quiz-a",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    later = _attempt(
        attempt_id="a1",
        quiz_id="quiz-a",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        labels=("A", "C"),
    )
    other = _attempt(
        attempt_id="b1",
        quiz_id="quiz-b",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    with SqliteQuizAttemptStore(db_path) as store:
        store.add(later)
        store.add(other)
        store.add(early)
        assert [item.attempt_id for item in store.list_for_quiz("quiz-a")] == ["a2", "a1"]
        assert [item.attempt_id for item in store.list_all()] == ["a2", "b1", "a1"]
        assert store.list_for_quiz("missing") == ()
        assert store.count() == 3


def test_delete_attempt(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    keep = _attempt(attempt_id="keep")
    drop = _attempt(attempt_id="drop", quiz_id="quiz-2")
    with SqliteQuizAttemptStore(db_path) as store:
        store.add(keep)
        store.add(drop)
        assert store.delete("drop") is True
        assert store.delete("drop") is False
        assert store.get("drop") is None
        assert store.get("keep") == keep
        assert store.count() == 1


def test_blank_attempt_id_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "attempts.sqlite"
    attempt = _attempt(attempt_id="   ")
    with SqliteQuizAttemptStore(db_path) as store:
        with pytest.raises(QuizAttemptStoreError, match="attempt_id"):
            store.add(attempt)
        assert store.count() == 0


def test_deserialize_question_results_defaults_missing_topic() -> None:
    encoded = json.dumps(
        [
            {
                "correct_label": "A",
                "explanation": "Stacks use LIFO.",
                "is_correct": True,
                "is_unanswered": False,
                "question_index": 0,
                "selected_label": "A",
                "sources": [
                    {
                        "chunk_id": "notes.pdf:p1:c1",
                        "citation_index": 1,
                        "page_number": 1,
                        "source_path": "notes.pdf",
                    }
                ],
            }
        ]
    )
    results = deserialize_question_results(encoded)
    assert results[0].topic == ""
