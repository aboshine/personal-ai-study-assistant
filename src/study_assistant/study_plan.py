"""Deterministic study plans from topic performance statistics.

This module does not call an LLM, persist data, or generate quizzes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.adaptive_quiz import EASY_ACCURACY_BELOW, HARD_ACCURACY_AT_LEAST
from study_assistant.quiz import Difficulty
from study_assistant.topic_tracking import TopicPerformance


@dataclass(frozen=True)
class StudyPlanItem:
    """One topic to study, ordered by need."""

    topic: str
    priority: int
    recommended_difficulty: Difficulty
    reason: str


@dataclass(frozen=True)
class StudyPlan:
    """A saved or generated sequence of study-plan items."""

    plan_id: str
    items: tuple[StudyPlanItem, ...]


def generate_study_plan(
    performances: Sequence[TopicPerformance] = (),
    *,
    available_topics: Sequence[str] = (),
) -> tuple[StudyPlanItem, ...]:
    """Build a study plan. Weaker and unpracticed topics come first."""
    stats = {_normalize(item.topic): item for item in performances if _normalize(item.topic)}
    names: list[str] = []
    seen: set[str] = set()
    for item in performances:
        name = _normalize(item.topic)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    for raw in available_topics:
        name = _normalize(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    ranked = sorted(names, key=lambda topic: _weakness_key(topic, stats.get(topic)))
    return tuple(
        _item(topic, priority=priority, performance=stats.get(topic))
        for priority, topic in enumerate(ranked, start=1)
    )


def _item(topic: str, *, priority: int, performance: TopicPerformance | None) -> StudyPlanItem:
    difficulty = _recommended_difficulty(performance)
    return StudyPlanItem(
        topic=topic,
        priority=priority,
        recommended_difficulty=difficulty,
        reason=_reason(performance, difficulty),
    )


def _recommended_difficulty(performance: TopicPerformance | None) -> Difficulty:
    if performance is None or performance.questions_answered == 0 or performance.accuracy < EASY_ACCURACY_BELOW:
        return "easy"
    if performance.accuracy < HARD_ACCURACY_AT_LEAST:
        return "medium"
    return "hard"


def _reason(performance: TopicPerformance | None, difficulty: Difficulty) -> str:
    if performance is None:
        return f"Unpracticed topic; start at {difficulty} difficulty."
    if performance.questions_answered == 0:
        return f"No scored answers yet; start at {difficulty} difficulty."
    if performance.accuracy < EASY_ACCURACY_BELOW:
        return f"Low accuracy ({performance.accuracy:.1f}%); practice at {difficulty} difficulty."
    if performance.accuracy < HARD_ACCURACY_AT_LEAST:
        return f"Moderate accuracy ({performance.accuracy:.1f}%); continue at {difficulty} difficulty."
    return f"High accuracy ({performance.accuracy:.1f}%); continue at {difficulty} difficulty."


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


def _normalize(value: str) -> str:
    return value.strip()
