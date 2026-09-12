"""Deterministic adaptive quiz topic and difficulty selection.

This module does not score quizzes, persist attempts, or retrieve chunks.
The LLM does not choose which topics are weak.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.llm import LLMClient
from study_assistant.quiz import (
    DEFAULT_DIFFICULTY,
    DEFAULT_QUESTION_COUNT,
    MAX_QUESTION_COUNT,
    Difficulty,
    Quiz,
    QuizGenerator,
)
from study_assistant.quiz_attempts import QuizAttempt
from study_assistant.topic_tracking import TopicPerformance, summarize_topic_performance
from study_assistant.vector_store import SimilarChunk

EASY_ACCURACY_BELOW = 50.0
HARD_ACCURACY_AT_LEAST = 80.0


@dataclass(frozen=True)
class AdaptiveQuizPlan:
    """Topic order, question mix, and difficulty for one adaptive quiz."""

    topics: tuple[str, ...]
    question_count: int
    difficulty: Difficulty
    question_allocation: tuple[tuple[str, int], ...]


def plan_adaptive_quiz(
    performances: Sequence[TopicPerformance] = (),
    *,
    question_count: int = DEFAULT_QUESTION_COUNT,
    available_topics: Sequence[str] = (),
    difficulty: Difficulty | None = None,
) -> AdaptiveQuizPlan:
    """Choose topics and difficulty from performance stats. Does not call an LLM."""
    count = _validate_question_count(question_count)
    ordered = _priority_topics(performances, available_topics)
    allocation = _allocate_questions(ordered, count)
    topics = tuple(topic for topic, _amount in allocation)
    selected_difficulty = difficulty if difficulty is not None else _select_difficulty(performances, topics)
    return AdaptiveQuizPlan(
        topics=topics,
        question_count=count,
        difficulty=selected_difficulty,
        question_allocation=allocation,
    )


def generate_adaptive_quiz(
    chunks: Sequence[SimilarChunk],
    *,
    llm: LLMClient,
    attempts: Sequence[QuizAttempt] = (),
    question_count: int = DEFAULT_QUESTION_COUNT,
    available_topics: Sequence[str] = (),
    difficulty: Difficulty | None = None,
    generator: QuizGenerator | None = None,
) -> Quiz:
    """Plan from topic stats, then generate via `QuizGenerator`."""
    plan = plan_adaptive_quiz(
        summarize_topic_performance(attempts),
        question_count=question_count,
        available_topics=available_topics,
        difficulty=difficulty,
    )
    quiz_generator = generator if generator is not None else QuizGenerator(llm)
    return quiz_generator.generate(
        chunks,
        question_count=plan.question_count,
        difficulty=plan.difficulty,
        topic_allocation=plan.question_allocation,
    )


def _priority_topics(
    performances: Sequence[TopicPerformance],
    available_topics: Sequence[str],
) -> tuple[str, ...]:
    stats = {_normalize_topic(item.topic): item for item in performances if _normalize_topic(item.topic)}
    names: list[str] = []
    seen: set[str] = set()
    candidates = [_normalize_topic(name) for name in available_topics]
    if any(candidates):
        source = candidates
    else:
        source = list(stats)
    for name in source:
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    ranked = sorted(names, key=lambda topic: _weakness_key(topic, stats.get(topic)))
    return tuple(ranked)


def _weakness_key(topic: str, item: TopicPerformance | None) -> tuple[float, int, int, str, str]:
    if item is None:
        return (0.0, 0, 0, topic.casefold(), topic)
    return (
        item.accuracy,
        item.questions_answered,
        item.attempt_count,
        topic.casefold(),
        topic,
    )


def _allocate_questions(topics: Sequence[str], question_count: int) -> tuple[tuple[str, int], ...]:
    if not topics:
        return ()
    counts = {topic: 0 for topic in topics}
    for index in range(question_count):
        counts[topics[index % len(topics)]] += 1
    return tuple((topic, counts[topic]) for topic in topics if counts[topic] > 0)


def _select_difficulty(
    performances: Sequence[TopicPerformance],
    topics: Sequence[str],
) -> Difficulty:
    if not topics:
        return DEFAULT_DIFFICULTY
    by_name = {_normalize_topic(item.topic): item for item in performances}
    weakest = by_name.get(topics[0])
    if weakest is None or weakest.questions_answered == 0 or weakest.accuracy < EASY_ACCURACY_BELOW:
        return "easy"
    if weakest.accuracy < HARD_ACCURACY_AT_LEAST:
        return "medium"
    return "hard"


def _normalize_topic(value: str) -> str:
    return value.strip()


def _validate_question_count(question_count: int) -> int:
    if question_count < 1:
        raise ValueError("question_count must be at least 1")
    if question_count > MAX_QUESTION_COUNT:
        raise ValueError(f"question_count must be at most {MAX_QUESTION_COUNT}")
    return question_count
