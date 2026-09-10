"""Deterministic scoring of submitted multiple-choice quiz answers.

This module does not generate quizzes, call an LLM, or touch storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.quiz import OPTION_LABELS, QuestionSource, Quiz

_VALID_LABELS = set(OPTION_LABELS)


class QuizEvaluationError(Exception):
    """Raised when submitted answers cannot be evaluated."""


@dataclass(frozen=True)
class SubmittedAnswer:
    """One user response. `selected_label` is None when the question is skipped."""

    question_index: int
    selected_label: str | None


@dataclass(frozen=True)
class QuestionResult:
    """Outcome for a single quiz question."""

    question_index: int
    selected_label: str | None
    correct_label: str
    is_correct: bool
    is_unanswered: bool
    explanation: str
    sources: tuple[QuestionSource, ...]


@dataclass(frozen=True)
class QuizEvaluation:
    """Aggregate score plus per-question results."""

    total_questions: int
    correct_count: int
    incorrect_count: int
    unanswered_count: int
    score_percent: float
    question_results: tuple[QuestionResult, ...]


def evaluate_quiz(quiz: Quiz, answers: Sequence[SubmittedAnswer]) -> QuizEvaluation:
    """Score `answers` against `quiz`. Unlisted questions count as unanswered."""
    total = len(quiz.questions)
    selected = _index_answers(answers, total)

    results: list[QuestionResult] = []
    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0

    for question_index, question in enumerate(quiz.questions):
        selected_label = selected.get(question_index)
        unanswered = selected_label is None
        is_correct = (not unanswered) and selected_label == question.correct_label
        if unanswered:
            unanswered_count += 1
        elif is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        results.append(
            QuestionResult(
                question_index=question_index,
                selected_label=selected_label,
                correct_label=question.correct_label,
                is_correct=is_correct,
                is_unanswered=unanswered,
                explanation=question.explanation,
                sources=question.sources,
            )
        )

    score_percent = 0.0 if total == 0 else (correct_count / total) * 100.0
    return QuizEvaluation(
        total_questions=total,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        score_percent=score_percent,
        question_results=tuple(results),
    )


def _index_answers(answers: Sequence[SubmittedAnswer], total: int) -> dict[int, str | None]:
    indexed: dict[int, str | None] = {}
    for answer in answers:
        if answer.question_index < 0 or answer.question_index >= total:
            raise QuizEvaluationError(
                f"Question index {answer.question_index} is out of range for a quiz with {total} question(s)"
            )
        if answer.question_index in indexed:
            raise QuizEvaluationError(f"Duplicate answer for question index {answer.question_index}")
        indexed[answer.question_index] = _normalize_label(answer.selected_label)
    return indexed


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    cleaned = label.strip().upper()
    if cleaned not in _VALID_LABELS:
        raise QuizEvaluationError(
            f"Invalid option label {label!r}; expected one of {', '.join(OPTION_LABELS)}"
        )
    return cleaned
