"""Unified in-memory study planning from existing topic, plan, and review logic.

This module does not score quizzes, generate LLM quizzes, or persist unless asked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from study_assistant.quiz import Difficulty
from study_assistant.quiz_attempts import QuizAttempt
from study_assistant.spaced_repetition import ReviewItem, schedule_reviews
from study_assistant.study_plan import StudyPlan, StudyPlanItem, generate_study_plan
from study_assistant.topic_tracking import TopicPerformance, summarize_topic_performance


@dataclass(frozen=True)
class StudyPlannerItem:
    """One planner row: priority, difficulty, review date, and combined reason."""

    topic: str
    priority: int
    recommended_difficulty: Difficulty
    next_review: datetime
    reason: str


@dataclass(frozen=True)
class UnifiedStudyPlan:
    """In-memory planner output. Convert to `StudyPlan` only when saving."""

    items: tuple[StudyPlannerItem, ...]

    def as_study_plan(self, plan_id: str) -> StudyPlan:
        """Map to the persistable `StudyPlan` shape (no review dates)."""
        return StudyPlan(
            plan_id=plan_id,
            items=tuple(
                StudyPlanItem(
                    topic=item.topic,
                    priority=item.priority,
                    recommended_difficulty=item.recommended_difficulty,
                    reason=item.reason,
                )
                for item in self.items
            ),
        )


class StudyPlanner:
    """Orchestrate topic stats, study-plan ranking, and review dates."""

    def plan(
        self,
        performances: Sequence[TopicPerformance] = (),
        *,
        attempts: Sequence[QuizAttempt] = (),
        available_topics: Sequence[str] = (),
        as_of: datetime | None = None,
    ) -> UnifiedStudyPlan:
        """Build one plan. Pass `attempts` to derive stats via topic tracking."""
        stats = summarize_topic_performance(attempts) if attempts else tuple(performances)
        plan_items = generate_study_plan(stats, available_topics=available_topics)
        reviews = schedule_reviews(stats, available_topics=available_topics, as_of=as_of)
        return UnifiedStudyPlan(items=_merge(plan_items, reviews))


def _merge(
    plan_items: Sequence[StudyPlanItem],
    reviews: Sequence[ReviewItem],
) -> tuple[StudyPlannerItem, ...]:
    by_review = {item.topic: item for item in reviews}
    combined: list[tuple[datetime, int, str, str, StudyPlanItem, ReviewItem]] = []
    for plan_item in plan_items:
        review = by_review[plan_item.topic]
        combined.append(
            (
                review.next_review,
                plan_item.priority,
                plan_item.topic.casefold(),
                plan_item.topic,
                plan_item,
                review,
            )
        )
    combined.sort()
    return tuple(
        StudyPlannerItem(
            topic=plan_item.topic,
            priority=index,
            recommended_difficulty=plan_item.recommended_difficulty,
            next_review=review.next_review,
            reason=f"{plan_item.reason} {review.reason}",
        )
        for index, (_when, _old, _fold, _name, plan_item, review) in enumerate(combined, start=1)
    )
