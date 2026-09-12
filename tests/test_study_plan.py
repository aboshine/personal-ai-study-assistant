from study_assistant.study_plan import StudyPlanItem, generate_study_plan
from study_assistant.topic_tracking import TopicPerformance


def _perf(
    topic: str,
    *,
    accuracy: float,
    questions_answered: int = 4,
    correct_count: int | None = None,
    attempt_count: int = 1,
) -> TopicPerformance:
    if correct_count is None:
        correct_count = round(accuracy / 100.0 * questions_answered)
    return TopicPerformance(
        topic=topic,
        attempt_count=attempt_count,
        questions_answered=questions_answered,
        correct_count=correct_count,
        accuracy=accuracy,
    )


def test_weak_topics_receive_higher_priority() -> None:
    plan = generate_study_plan(
        (
            _perf("Stacks", accuracy=90.0),
            _perf("Queues", accuracy=25.0),
            _perf("Heaps", accuracy=60.0),
        )
    )
    assert [item.topic for item in plan] == ["Queues", "Heaps", "Stacks"]
    assert [item.priority for item in plan] == [1, 2, 3]
    assert plan[0].priority < plan[1].priority < plan[2].priority
    assert plan[0].recommended_difficulty == "easy"
    assert plan[1].recommended_difficulty == "medium"
    assert plan[2].recommended_difficulty == "hard"


def test_unpracticed_topics_are_included_and_ranked_with_weakest() -> None:
    plan = generate_study_plan(
        (_perf("Stacks", accuracy=80.0),),
        available_topics=("Graphs", "Stacks"),
    )
    assert [item.topic for item in plan] == ["Graphs", "Stacks"]
    graphs = plan[0]
    assert graphs == StudyPlanItem(
        topic="Graphs",
        priority=1,
        recommended_difficulty="easy",
        reason="Unpracticed topic; start at easy difficulty.",
    )
    assert plan[1].topic == "Stacks"
    assert plan[1].priority == 2


def test_difficulty_mapping_is_correct() -> None:
    plan = generate_study_plan(
        (
            _perf("Queues", accuracy=20.0),
            _perf("Heaps", accuracy=60.0),
            _perf("Stacks", accuracy=90.0),
            _perf("Trees", accuracy=0.0, questions_answered=0, correct_count=0),
        )
    )
    by_topic = {item.topic: item for item in plan}
    assert by_topic["Queues"].recommended_difficulty == "easy"
    assert "Low accuracy" in by_topic["Queues"].reason
    assert by_topic["Heaps"].recommended_difficulty == "medium"
    assert "Moderate accuracy" in by_topic["Heaps"].reason
    assert by_topic["Stacks"].recommended_difficulty == "hard"
    assert "High accuracy" in by_topic["Stacks"].reason
    assert by_topic["Trees"].recommended_difficulty == "easy"
    assert by_topic["Trees"].reason == "No scored answers yet; start at easy difficulty."
    assert by_topic["Trees"].priority == 1


def test_ties_are_deterministic() -> None:
    tied = (
        _perf("Stacks", accuracy=50.0, questions_answered=2, correct_count=1),
        _perf("Queues", accuracy=50.0, questions_answered=2, correct_count=1),
    )
    first = generate_study_plan(tied)
    second = generate_study_plan(tuple(reversed(tied)))
    assert first == second
    assert [item.topic for item in first] == ["Queues", "Stacks"]
    assert [item.priority for item in first] == [1, 2]


def test_empty_and_no_history_input_works() -> None:
    assert generate_study_plan() == ()
    assert generate_study_plan(()) == ()
    plan = generate_study_plan((), available_topics=("Stacks", "Queues"))
    assert [item.topic for item in plan] == ["Queues", "Stacks"]
    assert all(item.recommended_difficulty == "easy" for item in plan)
    assert all("Unpracticed" in item.reason for item in plan)
    assert plan[0].priority == 1
    assert plan[1].priority == 2
