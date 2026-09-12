"""Deterministic per-topic performance from completed quiz attempts.

This module does not generate quizzes, call an LLM, or recommend what to study next.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.quiz_attempts import QuizAttempt


@dataclass(frozen=True)
class TopicPerformance:
    """Aggregated results for one topic across completed attempts."""

    topic: str
    attempt_count: int
    questions_answered: int
    correct_count: int
    accuracy: float


def summarize_topic_performance(attempts: Sequence[QuizAttempt]) -> tuple[TopicPerformance, ...]:
    """Aggregate topic stats from stored attempts. Unanswered questions are not scored."""
    attempt_ids: dict[str, set[str]] = {}
    answered: dict[str, int] = {}
    correct: dict[str, int] = {}

    for attempt in attempts:
        seen_in_attempt: set[str] = set()
        for result in attempt.question_results:
            topic = result.topic.strip()
            if not topic:
                continue
            seen_in_attempt.add(topic)
            if not result.is_unanswered:
                answered[topic] = answered.get(topic, 0) + 1
                if result.is_correct:
                    correct[topic] = correct.get(topic, 0) + 1
        for topic in seen_in_attempt:
            ids = attempt_ids.setdefault(topic, set())
            ids.add(attempt.attempt_id)

    summaries = [
        TopicPerformance(
            topic=topic,
            attempt_count=len(attempt_ids[topic]),
            questions_answered=answered.get(topic, 0),
            correct_count=correct.get(topic, 0),
            accuracy=_accuracy(correct.get(topic, 0), answered.get(topic, 0)),
        )
        for topic in attempt_ids
    ]
    summaries.sort(key=lambda item: (item.topic.casefold(), item.topic))
    return tuple(summaries)


def _accuracy(correct_count: int, questions_answered: int) -> float:
    if questions_answered == 0:
        return 0.0
    return (correct_count / questions_answered) * 100.0
