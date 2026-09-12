from datetime import datetime, timedelta, timezone

from study_assistant.spaced_repetition import ReviewItem, schedule_reviews
from study_assistant.topic_tracking import TopicPerformance

AS_OF = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _perf(
    topic: str,
    *,
    accuracy: float,
    questions_answered: int = 4,
    correct_count: int | None = None,
    attempt_count: int = 1,
) -> TopicPerformance:
    if correct_count is None:
        correct_count = round(accuracy / 100.0 * questions_answered) if questions_answered else 0
    return TopicPerformance(
        topic=topic,
        attempt_count=attempt_count,
        questions_answered=questions_answered,
        correct_count=correct_count,
        accuracy=accuracy,
    )


def test_unpracticed_topics_are_scheduled_first() -> None:
    items = schedule_reviews(
        (_perf("Stacks", accuracy=90.0),),
        available_topics=("Graphs", "Stacks"),
        as_of=AS_OF,
    )
    assert [item.topic for item in items] == ["Graphs", "Stacks"]
    graphs = items[0]
    assert graphs == ReviewItem(
        topic="Graphs",
        current_accuracy=0.0,
        interval=timedelta(days=0),
        next_review=AS_OF,
        reason="Unpracticed topic; review immediately.",
    )
    assert items[1].interval > graphs.interval


def test_weak_topics_get_shorter_intervals_than_stronger() -> None:
    items = schedule_reviews(
        (
            _perf("Stacks", accuracy=90.0),
            _perf("Queues", accuracy=25.0),
            _perf("Heaps", accuracy=60.0),
        ),
        as_of=AS_OF,
    )
    by_topic = {item.topic: item for item in items}
    assert by_topic["Queues"].interval < by_topic["Heaps"].interval < by_topic["Stacks"].interval
    assert [item.topic for item in items] == ["Queues", "Heaps", "Stacks"]


def test_stronger_topics_get_longer_intervals() -> None:
    weak, strong = schedule_reviews(
        (
            _perf("Queues", accuracy=20.0),
            _perf("Stacks", accuracy=90.0),
        ),
        as_of=AS_OF,
    )
    assert weak.topic == "Queues"
    assert strong.topic == "Stacks"
    assert strong.interval > weak.interval
    assert strong.next_review > weak.next_review


def test_results_are_deterministic() -> None:
    performances = (
        _perf("Stacks", accuracy=50.0, questions_answered=2, correct_count=1),
        _perf("Queues", accuracy=50.0, questions_answered=2, correct_count=1),
    )
    first = schedule_reviews(performances, as_of=AS_OF)
    second = schedule_reviews(tuple(reversed(performances)), as_of=AS_OF)
    assert first == second
    assert [item.topic for item in first] == ["Queues", "Stacks"]
    assert first[0].interval == first[1].interval


def test_multiple_topics_and_edge_accuracy_values() -> None:
    items = schedule_reviews(
        (
            _perf("Zero", accuracy=0.0),
            _perf("Perfect", accuracy=100.0),
            _perf("Unscored", accuracy=0.0, questions_answered=0, correct_count=0),
        ),
        as_of=AS_OF,
    )
    by_topic = {item.topic: item for item in items}
    assert by_topic["Unscored"].interval == timedelta(days=0)
    assert by_topic["Unscored"].next_review == AS_OF
    assert by_topic["Unscored"].reason == "No scored answers yet; review immediately."
    assert by_topic["Zero"].current_accuracy == 0.0
    assert by_topic["Zero"].interval == timedelta(days=1)
    assert by_topic["Zero"].next_review == AS_OF + timedelta(days=1)
    assert by_topic["Perfect"].current_accuracy == 100.0
    assert by_topic["Perfect"].interval == timedelta(days=7)
    assert by_topic["Perfect"].next_review == AS_OF + timedelta(days=7)
    assert [item.topic for item in items] == ["Unscored", "Zero", "Perfect"]


def test_empty_history() -> None:
    assert schedule_reviews() == ()
    assert schedule_reviews((), as_of=AS_OF) == ()
    items = schedule_reviews((), available_topics=("Stacks", "Queues"), as_of=AS_OF)
    assert [item.topic for item in items] == ["Queues", "Stacks"]
    assert all(item.interval == timedelta(days=0) for item in items)
    assert all(item.next_review == AS_OF for item in items)
    assert all(item.current_accuracy == 0.0 for item in items)
