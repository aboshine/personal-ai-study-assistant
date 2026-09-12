from datetime import datetime, timezone
from pathlib import Path

import pytest

from study_assistant.quiz import AnswerOption, Difficulty, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_attempts import QuizAttempt, QuizAttemptError, create_quiz_attempt
from study_assistant.quiz_evaluation import QuizEvaluation, SubmittedAnswer, evaluate_quiz


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


def _two_question_quiz(*, difficulty: Difficulty = "medium", quiz_id: str = "quiz-1") -> Quiz:
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
        difficulty=difficulty,
        quiz_id=quiz_id,
    )


def _evaluation(quiz: Quiz, *labels: str | None) -> QuizEvaluation:
    answers = tuple(SubmittedAnswer(index, label) for index, label in enumerate(labels))
    return evaluate_quiz(quiz, answers)


def test_create_attempt_from_quiz_and_evaluation() -> None:
    quiz = _two_question_quiz()
    evaluation = _evaluation(quiz, "A", "B")
    stamp = datetime(2026, 9, 10, 2, 30, tzinfo=timezone.utc)

    attempt = create_quiz_attempt(quiz, evaluation, timestamp=stamp)

    assert isinstance(attempt, QuizAttempt)
    assert attempt.quiz_id == "quiz-1"
    assert attempt.attempt_id
    assert attempt.timestamp == stamp
    assert attempt.evaluation is evaluation
    assert attempt.difficulty == "medium"
    assert attempt.question_results is evaluation.question_results
    assert attempt.question_results == evaluation.question_results


def test_from_quiz_classmethod_matches_factory() -> None:
    quiz = _two_question_quiz(quiz_id="quiz-classmethod")
    evaluation = _evaluation(quiz, "A", "C")
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    via_class = QuizAttempt.from_quiz(quiz, evaluation, timestamp=stamp)
    via_function = create_quiz_attempt(quiz, evaluation, timestamp=stamp)
    assert via_class.quiz_id == via_function.quiz_id
    assert via_class.timestamp == via_function.timestamp
    assert via_class.evaluation == via_function.evaluation
    assert via_class.difficulty == via_function.difficulty
    assert via_class.question_results == via_function.question_results
    assert via_class.attempt_id
    assert via_function.attempt_id
    assert via_class.attempt_id != via_function.attempt_id


def test_attempt_fields_include_per_question_results() -> None:
    quiz = _two_question_quiz()
    evaluation = _evaluation(quiz, "A", None)
    attempt = create_quiz_attempt(quiz, evaluation)

    assert len(attempt.question_results) == 2
    first, second = attempt.question_results
    assert first.question_index == 0
    assert first.is_correct is True
    assert second.is_unanswered is True
    assert attempt.evaluation.correct_count == 1
    assert attempt.evaluation.unanswered_count == 1
    assert attempt.evaluation.score_percent == 50.0


def test_default_timestamp_is_timezone_aware_utc() -> None:
    quiz = _two_question_quiz()
    evaluation = _evaluation(quiz, "A", "B")
    before = datetime.now(timezone.utc)
    attempt = create_quiz_attempt(quiz, evaluation)
    after = datetime.now(timezone.utc)
    assert attempt.timestamp.tzinfo is not None
    assert before <= attempt.timestamp <= after


def test_naive_timestamp_is_treated_as_utc() -> None:
    quiz = _two_question_quiz()
    evaluation = _evaluation(quiz, "A", "B")
    naive = datetime(2026, 9, 10, 12, 0, 0)
    attempt = create_quiz_attempt(quiz, evaluation, timestamp=naive)
    assert attempt.timestamp.tzinfo is timezone.utc
    assert attempt.timestamp.replace(tzinfo=None) == naive


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_attempt_preserves_quiz_difficulty(difficulty: Difficulty) -> None:
    quiz = _two_question_quiz(difficulty=difficulty, quiz_id=f"quiz-{difficulty}")
    evaluation = _evaluation(quiz, "A", "B")
    attempt = create_quiz_attempt(quiz, evaluation)
    assert attempt.difficulty == difficulty
    assert attempt.quiz_id == f"quiz-{difficulty}"


def test_generated_quiz_id_is_preserved_when_not_overridden() -> None:
    quiz = Quiz(
        questions=(
            _question(
                "What ordering does a stack use?",
                "A",
                "Stacks use LIFO.",
                (_source("notes.pdf:p1:c1"),),
            ),
        )
    )
    assert quiz.quiz_id
    evaluation = evaluate_quiz(quiz, (SubmittedAnswer(0, "A"),))
    attempt = create_quiz_attempt(quiz, evaluation)
    assert attempt.quiz_id == quiz.quiz_id


def test_blank_quiz_id_is_rejected() -> None:
    quiz = _two_question_quiz(quiz_id="   ")
    evaluation = _evaluation(quiz, "A", "B")
    with pytest.raises(QuizAttemptError, match="identifier"):
        create_quiz_attempt(quiz, evaluation)


def test_mismatched_question_count_is_rejected() -> None:
    quiz = _two_question_quiz()
    other = Quiz(
        questions=quiz.questions[:1],
        difficulty=quiz.difficulty,
        quiz_id=quiz.quiz_id,
    )
    evaluation = evaluate_quiz(other, (SubmittedAnswer(0, "A"),))
    with pytest.raises(QuizAttemptError, match="does not belong"):
        create_quiz_attempt(quiz, evaluation)


def test_evaluation_from_different_quiz_is_rejected() -> None:
    quiz = _two_question_quiz(quiz_id="original")
    other = Quiz(
        questions=(
            _question(
                "What ordering does a stack use?",
                "C",
                "Different correct answer.",
                (_source("notes.pdf:p1:c1", page=1),),
            ),
            quiz.questions[1],
        ),
        difficulty=quiz.difficulty,
        quiz_id="other",
    )
    evaluation = evaluate_quiz(other, (SubmittedAnswer(0, "C"), SubmittedAnswer(1, "B")))
    with pytest.raises(QuizAttemptError, match="does not belong"):
        create_quiz_attempt(quiz, evaluation)


def test_inconsistent_evaluation_counts_are_rejected() -> None:
    quiz = _two_question_quiz()
    evaluation = _evaluation(quiz, "A", "B")
    tampered = QuizEvaluation(
        total_questions=evaluation.total_questions,
        correct_count=0,
        incorrect_count=evaluation.incorrect_count,
        unanswered_count=evaluation.unanswered_count,
        score_percent=evaluation.score_percent,
        question_results=evaluation.question_results,
    )
    with pytest.raises(QuizAttemptError, match="inconsistent"):
        create_quiz_attempt(quiz, tampered)


def test_empty_quiz_attempt_is_allowed() -> None:
    quiz = Quiz(questions=(), quiz_id="empty")
    evaluation = evaluate_quiz(quiz, ())
    attempt = create_quiz_attempt(quiz, evaluation)
    assert attempt.quiz_id == "empty"
    assert attempt.question_results == ()
    assert attempt.evaluation.score_percent == 0.0
