"""In-memory records of completed quiz attempts.

This module does not score quizzes, persist to a database, or call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from study_assistant.quiz import Difficulty, Quiz
from study_assistant.quiz_evaluation import QuestionResult, QuizEvaluation


class QuizAttemptError(Exception):
    """Raised when a quiz attempt cannot be created from the given inputs."""


@dataclass(frozen=True)
class QuizAttempt:
    """One completed attempt: identity, time, score, and per-question outcomes."""

    quiz_id: str
    timestamp: datetime
    evaluation: QuizEvaluation
    difficulty: Difficulty
    question_results: tuple[QuestionResult, ...]
    attempt_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_quiz(
        cls,
        quiz: Quiz,
        evaluation: QuizEvaluation,
        *,
        timestamp: datetime | None = None,
    ) -> QuizAttempt:
        """Create an attempt from a `Quiz` and its `QuizEvaluation`."""
        return create_quiz_attempt(quiz, evaluation, timestamp=timestamp)


def create_quiz_attempt(
    quiz: Quiz,
    evaluation: QuizEvaluation,
    *,
    timestamp: datetime | None = None,
) -> QuizAttempt:
    """Record a completed attempt. Does not re-score answers."""
    _validate_evaluation_matches_quiz(quiz, evaluation)
    quiz_id = quiz.quiz_id.strip()
    if not quiz_id:
        raise QuizAttemptError("Quiz identifier must be a non-empty string")
    recorded_at = timestamp if timestamp is not None else datetime.now(timezone.utc)
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    return QuizAttempt(
        quiz_id=quiz_id,
        timestamp=recorded_at,
        evaluation=evaluation,
        difficulty=quiz.difficulty,
        question_results=evaluation.question_results,
        attempt_id=uuid4().hex,
    )


def _validate_evaluation_matches_quiz(quiz: Quiz, evaluation: QuizEvaluation) -> None:
    questions = quiz.questions
    results = evaluation.question_results
    if evaluation.total_questions != len(questions):
        raise QuizAttemptError("Evaluation does not belong to this quiz")
    if len(results) != len(questions):
        raise QuizAttemptError("Evaluation does not belong to this quiz")

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    for index, (question, result) in enumerate(zip(questions, results, strict=True)):
        if result.question_index != index:
            raise QuizAttemptError("Evaluation does not belong to this quiz")
        if result.correct_label != question.correct_label:
            raise QuizAttemptError("Evaluation does not belong to this quiz")
        if result.explanation != question.explanation:
            raise QuizAttemptError("Evaluation does not belong to this quiz")
        if result.sources != question.sources:
            raise QuizAttemptError("Evaluation does not belong to this quiz")
        if result.topic != question.topic:
            raise QuizAttemptError("Evaluation does not belong to this quiz")
        if result.is_unanswered:
            unanswered_count += 1
        elif result.is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    if (
        evaluation.correct_count != correct_count
        or evaluation.incorrect_count != incorrect_count
        or evaluation.unanswered_count != unanswered_count
    ):
        raise QuizAttemptError("Evaluation result is inconsistent")
    expected_percent = 0.0 if not questions else (correct_count / len(questions)) * 100.0
    if evaluation.score_percent != expected_percent:
        raise QuizAttemptError("Evaluation result is inconsistent")
