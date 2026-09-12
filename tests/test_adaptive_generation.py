import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from study_assistant.adaptive_quiz import generate_adaptive_quiz
from study_assistant.llm.base import LLMClient
from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizGenerator, QuizQuestion
from study_assistant.quiz_attempts import create_quiz_attempt
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz
from study_assistant.vector_store import SimilarChunk


class FakeLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RecordingQuizGenerator(QuizGenerator):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm)
        self.calls: list[dict[str, Any]] = []

    def generate(self, chunks, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return super().generate(chunks, **kwargs)


def _chunk() -> SimilarChunk:
    return SimilarChunk(
        chunk_id="notes.pdf:p1:c1",
        source_path=Path("notes.pdf"),
        page_number=1,
        text="A stack uses LIFO ordering. A queue uses FIFO ordering.",
        similarity=0.9,
    )


def _quiz_json(question_count: int) -> str:
    questions = [
        {
            "question": f"Question {index}?",
            "options": [
                {"label": "A", "text": f"opt-a-{index}"},
                {"label": "B", "text": f"opt-b-{index}"},
                {"label": "C", "text": f"opt-c-{index}"},
                {"label": "D", "text": f"opt-d-{index}"},
            ],
            "correct_label": "A",
            "explanation": f"Because {index}.",
            "topic": "Queues",
            "source_citations": [1],
        }
        for index in range(question_count)
    ]
    return json.dumps({"questions": questions})


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


def _history_attempts():
    quiz = Quiz(
        questions=(
            _question("Stack ordering?", "A", "Stacks"),
            _question("Queue ordering?", "B", "Queues"),
        ),
        quiz_id="history-quiz",
    )
    evaluation = evaluate_quiz(quiz, (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")))
    attempt = create_quiz_attempt(
        quiz,
        evaluation,
        timestamp=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )
    return (attempt,)


def test_weak_topics_and_difficulty_reach_quiz_generator() -> None:
    llm = FakeLLM(_quiz_json(3))
    generator = RecordingQuizGenerator(llm)
    quiz = generate_adaptive_quiz(
        (_chunk(),),
        llm=llm,
        generator=generator,
        attempts=_history_attempts(),
        question_count=3,
    )

    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["question_count"] == 3
    assert call["difficulty"] == "easy"
    assert call["topic_allocation"][0][0] == "Queues"
    assert call["topic_allocation"] == (("Queues", 2), ("Stacks", 1))
    prompt = llm.prompts[0]
    assert "Prioritize these topics (weakest first): Queues, Stacks." in prompt
    assert "Question allocation: Queues: 2; Stacks: 1." in prompt
    assert "Difficulty: easy." in prompt
    assert quiz.difficulty == "easy"
    assert len(quiz.questions) == 3
    assert quiz.questions[0].sources[0].chunk_id == "notes.pdf:p1:c1"


def test_no_history_fallback_reaches_quiz_generator() -> None:
    llm = FakeLLM(_quiz_json(2))
    generator = RecordingQuizGenerator(llm)
    quiz = generate_adaptive_quiz(
        (_chunk(),),
        llm=llm,
        generator=generator,
        attempts=(),
        question_count=2,
    )

    assert generator.calls[0]["question_count"] == 2
    assert generator.calls[0]["difficulty"] == "medium"
    assert generator.calls[0]["topic_allocation"] == ()
    assert "Prioritize these topics" not in llm.prompts[0]
    assert "Difficulty: medium." in llm.prompts[0]
    assert quiz.difficulty == "medium"
    assert len(quiz.questions) == 2


def test_question_count_is_preserved_end_to_end() -> None:
    llm = FakeLLM(_quiz_json(4))
    generator = RecordingQuizGenerator(llm)
    quiz = generate_adaptive_quiz(
        (_chunk(),),
        llm=llm,
        generator=generator,
        attempts=_history_attempts(),
        question_count=4,
    )
    assert generator.calls[0]["question_count"] == 4
    assert sum(count for _topic, count in generator.calls[0]["topic_allocation"]) == 4
    assert len(quiz.questions) == 4


def test_end_to_end_selection_is_deterministic() -> None:
    attempts = _history_attempts()
    first_llm = FakeLLM(_quiz_json(3))
    second_llm = FakeLLM(_quiz_json(3))
    first = RecordingQuizGenerator(first_llm)
    second = RecordingQuizGenerator(second_llm)
    generate_adaptive_quiz((_chunk(),), llm=first_llm, generator=first, attempts=attempts, question_count=3)
    generate_adaptive_quiz((_chunk(),), llm=second_llm, generator=second, attempts=attempts, question_count=3)
    assert first.calls == second.calls
    assert first_llm.prompts[0] == second_llm.prompts[0]
