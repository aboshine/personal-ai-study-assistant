from datetime import datetime, timedelta, timezone
from pathlib import Path

from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_attempts import create_quiz_attempt
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz
from study_assistant.spaced_repetition import schedule_reviews
from study_assistant.study_plan import generate_study_plan
from study_assistant.study_planner import StudyPlanner, UnifiedStudyPlan
from study_assistant.topic_tracking import TopicPerformance, summarize_topic_performance

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


def _question(text: str, correct_label: str, topic: str) -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=tuple(
            AnswerOption(label=label, text=word)
            for label, word in zip("ABCD", ("LIFO", "FIFO", "Heap", "Graph"), strict=True)
        ),
        correct_label=correct_label,
        explanation=f"{text} answer.",
        sources=(
            QuestionSource(
                citation_index=1,
                chunk_id="notes.pdf:p1:c1",
                source_path=Path("notes.pdf"),
                page_number=1,
            ),
        ),
        topic=topic,
    )


def test_complete_planning_flow_from_attempts() -> None:
    quiz = Quiz(
        questions=(
            _question("Stack ordering?", "A", "Stacks"),
            _question("Queue ordering?", "B", "Queues"),
        ),
        quiz_id="history",
    )
    evaluation = evaluate_quiz(quiz, (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")))
    attempt = create_quiz_attempt(quiz, evaluation, timestamp=AS_OF)
    stats = summarize_topic_performance((attempt,))
    expected_plan = generate_study_plan(stats)
    expected_reviews = schedule_reviews(stats, as_of=AS_OF)

    unified = StudyPlanner().plan(attempts=(attempt,), as_of=AS_OF)

    assert isinstance(unified, UnifiedStudyPlan)
    assert [item.topic for item in unified.items] == [item.topic for item in expected_reviews]
    by_plan = {item.topic: item for item in expected_plan}
    by_review = {item.topic: item for item in expected_reviews}
    for item in unified.items:
        assert item.recommended_difficulty == by_plan[item.topic].recommended_difficulty
        assert item.next_review == by_review[item.topic].next_review
        assert by_plan[item.topic].reason in item.reason
        assert by_review[item.topic].reason in item.reason


def test_weak_topics_before_strong_and_unpracticed_first() -> None:
    unified = StudyPlanner().plan(
        (
            _perf("Stacks", accuracy=90.0),
            _perf("Queues", accuracy=25.0),
        ),
        available_topics=("Graphs", "Stacks", "Queues"),
        as_of=AS_OF,
    )
    assert [item.topic for item in unified.items] == ["Graphs", "Queues", "Stacks"]
    assert [item.priority for item in unified.items] == [1, 2, 3]
    graphs, queues, stacks = unified.items
    assert graphs.recommended_difficulty == "easy"
    assert graphs.next_review == AS_OF
    assert "Unpracticed" in graphs.reason
    assert queues.recommended_difficulty == "easy"
    assert queues.next_review == AS_OF + timedelta(days=2)
    assert stacks.recommended_difficulty == "hard"
    assert stacks.next_review > queues.next_review


def test_review_dates_and_difficulty_come_from_existing_algorithms() -> None:
    performances = (
        _perf("Heaps", accuracy=60.0),
        _perf("Queues", accuracy=20.0),
        _perf("Stacks", accuracy=100.0),
    )
    unified = StudyPlanner().plan(performances, as_of=AS_OF)
    plan_items = generate_study_plan(performances)
    reviews = schedule_reviews(performances, as_of=AS_OF)
    by_plan = {item.topic: item for item in plan_items}
    by_review = {item.topic: item for item in reviews}
    assert [item.topic for item in unified.items] == ["Queues", "Heaps", "Stacks"]
    for item in unified.items:
        assert item.next_review == by_review[item.topic].next_review
        assert item.recommended_difficulty == by_plan[item.topic].recommended_difficulty


def test_ordering_is_deterministic() -> None:
    performances = (
        _perf("Stacks", accuracy=50.0, questions_answered=2, correct_count=1),
        _perf("Queues", accuracy=50.0, questions_answered=2, correct_count=1),
    )
    planner = StudyPlanner()
    first = planner.plan(performances, as_of=AS_OF)
    second = planner.plan(tuple(reversed(performances)), as_of=AS_OF)
    assert first == second
    assert [item.topic for item in first.items] == ["Queues", "Stacks"]


def test_empty_and_no_history() -> None:
    planner = StudyPlanner()
    assert planner.plan(as_of=AS_OF).items == ()
    unified = planner.plan((), available_topics=("Stacks", "Queues"), as_of=AS_OF)
    assert [item.topic for item in unified.items] == ["Queues", "Stacks"]
    assert all(item.next_review == AS_OF for item in unified.items)
    assert all(item.recommended_difficulty == "easy" for item in unified.items)
    persistable = unified.as_study_plan("plan-empty-history")
    assert persistable.plan_id == "plan-empty-history"
    assert [item.topic for item in persistable.items] == ["Queues", "Stacks"]
    assert persistable.items[0].priority == 1
