from datetime import datetime, timezone
from pathlib import Path

from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore
from study_assistant.quiz_attempts import create_quiz_attempt
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz
from study_assistant.topic_tracking import TopicPerformance, summarize_topic_performance


def _options() -> tuple[AnswerOption, ...]:
    return tuple(AnswerOption(label=label, text=text) for label, text in zip("ABCD", ("LIFO", "FIFO", "Heap", "Graph"), strict=True))


def _source(chunk_id: str = "notes.pdf:p1:c1") -> QuestionSource:
    return QuestionSource(
        citation_index=1,
        chunk_id=chunk_id,
        source_path=Path("notes.pdf"),
        page_number=1,
    )


def _question(text: str, correct_label: str, topic: str) -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=_options(),
        correct_label=correct_label,
        explanation=f"{text} answer.",
        sources=(_source(),),
        topic=topic,
    )


def _quiz(*questions: QuizQuestion, quiz_id: str = "quiz-1") -> Quiz:
    return Quiz(questions=questions, quiz_id=quiz_id)


def _attempt(
    quiz: Quiz,
    *labels: str | None,
    attempt_id: str,
    timestamp: datetime | None = None,
):
    evaluation = evaluate_quiz(
        quiz,
        tuple(SubmittedAnswer(index, label) for index, label in enumerate(labels)),
    )
    attempt = create_quiz_attempt(
        quiz,
        evaluation,
        timestamp=timestamp or datetime(2026, 9, 10, tzinfo=timezone.utc),
    )
    return attempt.__class__(
        quiz_id=attempt.quiz_id,
        timestamp=attempt.timestamp,
        evaluation=attempt.evaluation,
        difficulty=attempt.difficulty,
        question_results=attempt.question_results,
        attempt_id=attempt_id,
    )


def test_empty_attempts_yield_no_topics() -> None:
    assert summarize_topic_performance(()) == ()


def test_multiple_topics_are_aggregated_and_sorted() -> None:
    quiz = _quiz(
        _question("Stack ordering?", "A", "Stacks"),
        _question("Queue ordering?", "B", "Queues"),
        _question("Heap shape?", "C", "Heaps"),
    )
    attempt = _attempt(quiz, "A", "A", "C", attempt_id="a1")
    heaps, queues, stacks = summarize_topic_performance((attempt,))
    assert [item.topic for item in (heaps, queues, stacks)] == ["Heaps", "Queues", "Stacks"]
    assert heaps == TopicPerformance("Heaps", 1, 1, 1, 100.0)
    assert queues == TopicPerformance("Queues", 1, 1, 0, 0.0)
    assert stacks == TopicPerformance("Stacks", 1, 1, 1, 100.0)


def test_repeated_attempts_accumulate_counts() -> None:
    quiz = _quiz(
        _question("Stack ordering?", "A", "Stacks"),
        _question("Queue ordering?", "B", "Queues"),
    )
    first = _attempt(quiz, "A", "B", attempt_id="a1")
    second = _attempt(quiz, "C", "B", attempt_id="a2")
    queues, stacks = summarize_topic_performance((first, second))
    assert stacks.topic == "Stacks"
    assert stacks.attempt_count == 2
    assert stacks.questions_answered == 2
    assert stacks.correct_count == 1
    assert stacks.accuracy == 50.0
    assert queues.attempt_count == 2
    assert queues.questions_answered == 2
    assert queues.correct_count == 2
    assert queues.accuracy == 100.0


def test_unanswered_questions_are_excluded_from_accuracy() -> None:
    quiz = _quiz(
        _question("Stack ordering?", "A", "Stacks"),
        _question("Another stack question?", "A", "Stacks"),
        _question("Queue ordering?", "B", "Queues"),
    )
    attempt = _attempt(quiz, "A", None, None, attempt_id="a1")
    queues, stacks = summarize_topic_performance((attempt,))
    assert stacks.attempt_count == 1
    assert stacks.questions_answered == 1
    assert stacks.correct_count == 1
    assert stacks.accuracy == 100.0
    assert queues.attempt_count == 1
    assert queues.questions_answered == 0
    assert queues.correct_count == 0
    assert queues.accuracy == 0.0


def test_questions_without_topic_are_ignored() -> None:
    quiz = _quiz(
        _question("Stack ordering?", "A", "Stacks"),
        _question("Untagged?", "B", ""),
    )
    attempt = _attempt(quiz, "A", "A", attempt_id="a1")
    (stacks,) = summarize_topic_performance((attempt,))
    assert stacks.topic == "Stacks"
    assert stacks.questions_answered == 1


def test_aggregation_from_stored_attempts_is_deterministic(tmp_path: Path) -> None:
    quiz = _quiz(
        _question("Stack ordering?", "A", "Stacks"),
        _question("Queue ordering?", "B", "Queues"),
    )
    first = _attempt(
        quiz,
        "A",
        "C",
        attempt_id="a1",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    second = _attempt(
        quiz,
        "A",
        "B",
        attempt_id="a2",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    db_path = tmp_path / "attempts.sqlite"
    with SqliteQuizAttemptStore(db_path) as writer:
        writer.add(first)
        writer.add(second)

    with SqliteQuizAttemptStore(db_path) as reader:
        from_store = summarize_topic_performance(reader.list_all())
        again = summarize_topic_performance(reader.list_all())

    assert from_store == again
    assert from_store == summarize_topic_performance((second, first))
    queues, stacks = from_store
    assert stacks.correct_count == 2
    assert stacks.questions_answered == 2
    assert queues.correct_count == 1
    assert queues.accuracy == 50.0
