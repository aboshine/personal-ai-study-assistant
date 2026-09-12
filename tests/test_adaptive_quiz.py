from pathlib import Path

import pytest

from study_assistant.adaptive_quiz import (
    AdaptiveQuizPlan,
    generate_adaptive_quiz,
    plan_adaptive_quiz,
)
from study_assistant.llm.base import LLMClient
from study_assistant.quiz import build_quiz_prompt
from study_assistant.topic_tracking import TopicPerformance
from study_assistant.vector_store import SimilarChunk


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


class FakeLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _chunk() -> SimilarChunk:
    return SimilarChunk(
        chunk_id="notes.pdf:p1:c1",
        source_path=Path("notes.pdf"),
        page_number=1,
        text="A stack uses LIFO ordering. A queue uses FIFO ordering.",
        similarity=0.9,
    )


def test_weak_topics_are_prioritized() -> None:
    plan = plan_adaptive_quiz(
        (
            _perf("Stacks", accuracy=90.0),
            _perf("Queues", accuracy=25.0),
            _perf("Heaps", accuracy=60.0),
        ),
        question_count=3,
    )
    assert plan.topics[0] == "Queues"
    assert plan.topics == ("Queues", "Heaps", "Stacks")
    assert plan.question_allocation[0] == ("Queues", 1)


def test_stronger_topics_receive_lower_priority_and_fewer_questions() -> None:
    plan = plan_adaptive_quiz(
        (
            _perf("Stacks", accuracy=100.0),
            _perf("Queues", accuracy=0.0, questions_answered=2, correct_count=0),
        ),
        question_count=3,
    )
    assert plan.topics == ("Queues", "Stacks")
    assert plan.question_allocation == (("Queues", 2), ("Stacks", 1))
    assert plan.question_count == 3


def test_ties_are_deterministic() -> None:
    tied = (
        _perf("Stacks", accuracy=50.0, questions_answered=2, correct_count=1),
        _perf("Queues", accuracy=50.0, questions_answered=2, correct_count=1),
    )
    first = plan_adaptive_quiz(tied, question_count=2)
    second = plan_adaptive_quiz(tuple(reversed(tied)), question_count=2)
    assert first == second
    assert first.topics == ("Queues", "Stacks")


def test_no_history_uses_medium_difficulty_and_requested_count() -> None:
    plan = plan_adaptive_quiz((), question_count=4)
    assert plan == AdaptiveQuizPlan(
        topics=(),
        question_count=4,
        difficulty="medium",
        question_allocation=(),
    )


def test_no_history_with_available_topics_is_alphabetical_and_easy() -> None:
    plan = plan_adaptive_quiz(
        (),
        question_count=3,
        available_topics=("Stacks", "Queues", "Heaps"),
    )
    assert plan.topics == ("Heaps", "Queues", "Stacks")
    assert plan.difficulty == "easy"
    assert sum(count for _topic, count in plan.question_allocation) == 3


def test_requested_question_count_is_respected() -> None:
    plan = plan_adaptive_quiz(
        (_perf("Queues", accuracy=10.0), _perf("Stacks", accuracy=90.0)),
        question_count=5,
    )
    assert plan.question_count == 5
    assert sum(count for _topic, count in plan.question_allocation) == 5
    with pytest.raises(ValueError, match="at least 1"):
        plan_adaptive_quiz((), question_count=0)
    with pytest.raises(ValueError, match="at most"):
        plan_adaptive_quiz((), question_count=21)


def test_difficulty_is_selected_from_weakest_topic() -> None:
    easy = plan_adaptive_quiz((_perf("Queues", accuracy=20.0),), question_count=1)
    medium = plan_adaptive_quiz((_perf("Stacks", accuracy=60.0),), question_count=1)
    hard = plan_adaptive_quiz((_perf("Heaps", accuracy=90.0),), question_count=1)
    mixed = plan_adaptive_quiz(
        (_perf("Queues", accuracy=20.0), _perf("Heaps", accuracy=95.0)),
        question_count=2,
    )
    unanswered = plan_adaptive_quiz(
        (_perf("Graphs", accuracy=0.0, questions_answered=0, correct_count=0),),
        question_count=1,
    )
    override = plan_adaptive_quiz(
        (_perf("Queues", accuracy=20.0),),
        question_count=1,
        difficulty="hard",
    )
    assert easy.difficulty == "easy"
    assert medium.difficulty == "medium"
    assert hard.difficulty == "hard"
    assert mixed.topics[0] == "Queues"
    assert mixed.difficulty == "easy"
    assert unanswered.difficulty == "easy"
    assert override.difficulty == "hard"


def test_generate_adaptive_quiz_uses_plan_without_llm_topic_choice() -> None:
    llm = FakeLLM(
        '{"questions":[{"question":"What ordering does a queue use?",'
        '"options":[{"label":"A","text":"LIFO"},{"label":"B","text":"FIFO"},'
        '{"label":"C","text":"Heap"},{"label":"D","text":"Graph"}],'
        '"correct_label":"B","explanation":"Queues are FIFO.","topic":"Queues",'
        '"source_citations":[1]}]}'
    )
    quiz = generate_adaptive_quiz(
        (_chunk(),),
        llm=llm,
        attempts=(),
        question_count=1,
        available_topics=("Queues",),
    )
    assert quiz.difficulty == "easy"
    assert len(quiz.questions) == 1
    assert "Prioritize these topics (weakest first): Queues." in llm.prompts[0]
    assert "Difficulty: easy." in llm.prompts[0]


def test_prompt_omits_allocation_when_empty() -> None:
    prompt = build_quiz_prompt((_chunk(),), question_count=1)
    assert "Prioritize these topics" not in prompt
    assert "Question allocation" not in prompt
