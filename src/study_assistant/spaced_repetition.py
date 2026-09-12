"""Deterministic next-review dates from topic learning history.

This module does not call an LLM, persist a schedule, or implement SM-2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from study_assistant.topic_tracking import TopicPerformance

MIN_PRACTICED_INTERVAL_DAYS = 1
MAX_INTERVAL_DAYS = 7


@dataclass(frozen=True)
class ReviewItem:
    """One topic's next review, derived from accuracy and practice history."""

    topic: str
    current_accuracy: float
    interval: timedelta
    next_review: datetime
    reason: str


def schedule_reviews(
    performances: Sequence[TopicPerformance] = (),
    *,
    available_topics: Sequence[str] = (),
    as_of: datetime | None = None,
) -> tuple[ReviewItem, ...]:
    """Schedule reviews. Weaker and unpracticed topics return sooner."""
    moment = _as_of(as_of)
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

    items = tuple(_review_item(topic, stats.get(topic), moment) for topic in names)
    return tuple(
        sorted(
            items,
            key=lambda item: (item.interval, item.next_review, item.topic.casefold(), item.topic),
        )
    )


def _review_item(topic: str, performance: TopicPerformance | None, as_of: datetime) -> ReviewItem:
    days = _interval_days(performance)
    accuracy = 0.0 if performance is None else performance.accuracy
    interval = timedelta(days=days)
    return ReviewItem(
        topic=topic,
        current_accuracy=accuracy,
        interval=interval,
        next_review=as_of + interval,
        reason=_reason(performance, days),
    )


def _interval_days(performance: TopicPerformance | None) -> int:
    if performance is None or performance.questions_answered == 0:
        return 0
    span = MAX_INTERVAL_DAYS - MIN_PRACTICED_INTERVAL_DAYS
    extra = int(performance.accuracy * span // 100)
    return MIN_PRACTICED_INTERVAL_DAYS + extra


def _reason(performance: TopicPerformance | None, days: int) -> str:
    if performance is None:
        return "Unpracticed topic; review immediately."
    if performance.questions_answered == 0:
        return "No scored answers yet; review immediately."
    if days == 1:
        return f"Accuracy {performance.accuracy:.1f}%; review in 1 day."
    return f"Accuracy {performance.accuracy:.1f}%; review in {days} days."


def _as_of(value: datetime | None) -> datetime:
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize(value: str) -> str:
    return value.strip()
