from __future__ import annotations

import json
from pathlib import Path

import pytest

from study_assistant.llm.base import LLMClient
from study_assistant.quiz import (
    DEFAULT_DIFFICULTY,
    DIFFICULTY_LEVELS,
    Quiz,
    QuizGenerationError,
    QuizGenerator,
    build_quiz_prompt,
    generate_quiz,
    parse_quiz_response,
)
from study_assistant.vector_store import SimilarChunk


class FakeLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "notes.pdf",
    page: int = 1,
    similarity: float = 0.9,
) -> SimilarChunk:
    return SimilarChunk(
        chunk_id=chunk_id,
        source_path=Path(source),
        page_number=page,
        text=text,
        similarity=similarity,
    )


def _option_payloads(texts: tuple[str, str, str, str]) -> list[dict[str, str]]:
    return [{"label": label, "text": text} for label, text in zip("ABCD", texts, strict=True)]


def _quiz_json(
    *,
    questions: list[dict[str, object]] | None = None,
    question: str = "What discipline uses LIFO ordering?",
    options: tuple[str, str, str, str] = ("Stack", "Queue", "Heap", "Graph"),
    correct_label: str = "A",
    explanation: str = "Stacks are LIFO.",
    source_citations: list[int] | None = None,
) -> str:
    if questions is None:
        questions = [
            {
                "question": question,
                "options": _option_payloads(options),
                "correct_label": correct_label,
                "explanation": explanation,
                "source_citations": source_citations if source_citations is not None else [1],
            }
        ]
    return json.dumps({"questions": questions})


def test_generate_quiz_parses_valid_multiple_choice() -> None:
    chunks = (
        _chunk("notes.pdf:p1:c1", "A stack uses LIFO ordering.", page=1),
        _chunk("ds.pdf:p3:c1", "A queue uses FIFO ordering.", source="ds.pdf", page=3),
    )
    llm = FakeLLM(
        _quiz_json(
            questions=[
                {
                    "question": "What ordering does a stack use?",
                    "options": _option_payloads(("LIFO", "FIFO", "Random", "Sorted")),
                    "correct_label": "A",
                    "explanation": "The notes define stacks as LIFO.",
                    "source_citations": [1],
                },
                {
                    "question": "What ordering does a queue use?",
                    "options": _option_payloads(("LIFO", "FIFO", "Priority", "None")),
                    "correct_label": "B",
                    "explanation": "The data-structures notes define queues as FIFO.",
                    "source_citations": [2],
                },
            ]
        )
    )

    quiz = QuizGenerator(llm).generate(chunks, question_count=2)

    assert len(quiz.questions) == 2
    first, second = quiz.questions
    assert first.question == "What ordering does a stack use?"
    assert [option.label for option in first.options] == ["A", "B", "C", "D"]
    assert len(first.options) == 4
    assert first.correct_label == "A"
    assert first.correct_label in {option.label for option in first.options}
    assert first.options[0].text == "LIFO"
    assert first.explanation == "The notes define stacks as LIFO."
    assert first.sources[0].citation_index == 1
    assert first.sources[0].chunk_id == "notes.pdf:p1:c1"
    assert first.sources[0].source_path == Path("notes.pdf")
    assert first.sources[0].page_number == 1
    assert second.correct_label == "B"
    assert second.sources[0].source_path == Path("ds.pdf")
    assert second.sources[0].page_number == 3
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "A stack uses LIFO ordering." in prompt
    assert "A queue uses FIFO ordering." in prompt
    assert "exactly 4 options" in prompt
    assert "Do not invent facts" in prompt
    assert "Invented outside fact" not in prompt
    assert quiz.difficulty == DEFAULT_DIFFICULTY
    assert "Difficulty: medium." in prompt


@pytest.mark.parametrize("difficulty", DIFFICULTY_LEVELS)
def test_generate_quiz_accepts_each_difficulty(difficulty: str) -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json())
    quiz = QuizGenerator(llm).generate(chunks, question_count=1, difficulty=difficulty)
    assert quiz.difficulty == difficulty
    assert f"Difficulty: {difficulty}." in llm.prompts[0]


def test_generate_quiz_defaults_to_medium() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json())
    quiz = QuizGenerator(llm).generate(chunks, question_count=1)
    assert quiz.difficulty == "medium"
    assert DEFAULT_DIFFICULTY == "medium"


def test_invalid_difficulty_is_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json())
    with pytest.raises(ValueError, match="easy, medium, hard"):
        QuizGenerator(llm).generate(chunks, question_count=1, difficulty="expert")
    with pytest.raises(ValueError, match="easy, medium, hard"):
        build_quiz_prompt(chunks, question_count=1, difficulty="nightmare")
    with pytest.raises(ValueError, match="easy, medium, hard"):
        parse_quiz_response(_quiz_json(), chunks, question_count=1, difficulty="")
    assert llm.prompts == []


def test_prompt_propagates_difficulty() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Only this sentence is allowed."),)
    easy = build_quiz_prompt(chunks, question_count=1, difficulty="easy")
    hard = build_quiz_prompt(chunks, question_count=1, difficulty="hard")
    medium = build_quiz_prompt(chunks, question_count=1)
    assert "Difficulty: easy." in easy
    assert "direct recall" in easy
    assert "Difficulty: hard." in hard
    assert "comparison or inference" in hard
    assert "Difficulty: medium." in medium
    assert "easy" not in hard.split("Difficulty:", 1)[1].split("\n", 1)[0]


def test_parse_quiz_response_stores_requested_difficulty() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    quiz = parse_quiz_response(_quiz_json(), chunks, question_count=1, difficulty="hard")
    assert quiz.difficulty == "hard"
    defaulted = parse_quiz_response(_quiz_json(), chunks, question_count=1)
    assert defaulted.difficulty == "medium"


def test_quiz_construction_defaults_difficulty_for_backward_compatibility() -> None:
    quiz = Quiz(questions=())
    assert quiz.difficulty == "medium"


def test_parse_quiz_response_defaults_missing_topic() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    quiz = parse_quiz_response(_quiz_json(), chunks, question_count=1)
    assert quiz.questions[0].topic == ""


def test_parse_quiz_response_stores_topic_when_present() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    payload = json.loads(_quiz_json())
    payload["questions"][0]["topic"] = "  Stacks  "
    quiz = parse_quiz_response(json.dumps(payload), chunks, question_count=1)
    assert quiz.questions[0].topic == "Stacks"


def test_prompt_asks_for_topic_labels() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    prompt = build_quiz_prompt(chunks, question_count=1)
    assert "short topic label" in prompt
    assert '"topic":"..."' in prompt


def test_prompt_contains_only_supplied_context() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Only this sentence is allowed."),)
    prompt = build_quiz_prompt(chunks, question_count=1)
    assert "Only this sentence is allowed." in prompt
    assert "secret unused material" not in prompt
    assert "[1] notes.pdf, page 1" in prompt


def test_empty_context_does_not_call_llm() -> None:
    llm = FakeLLM(_quiz_json())
    with pytest.raises(QuizGenerationError, match="without retrieved study material"):
        QuizGenerator(llm).generate((), question_count=1)
    assert llm.prompts == []


def test_invalid_question_counts_are_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json())
    with pytest.raises(ValueError, match="at least 1"):
        QuizGenerator(llm).generate(chunks, question_count=0)
    with pytest.raises(ValueError, match="at most"):
        QuizGenerator(llm).generate(chunks, question_count=21)
    assert llm.prompts == []


def test_malformed_json_is_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM("this is not json")
    with pytest.raises(QuizGenerationError, match="not valid JSON"):
        generate_quiz(chunks, llm=llm, question_count=1)


def test_wrong_question_count_in_llm_output_is_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json())
    with pytest.raises(QuizGenerationError, match="Expected 2 question"):
        QuizGenerator(llm).generate(chunks, question_count=2)


def test_not_four_options_is_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    payload = json.loads(_quiz_json())
    payload["questions"][0]["options"] = [{"label": "A", "text": "one"}, {"label": "B", "text": "two"}]
    llm = FakeLLM(json.dumps(payload))
    with pytest.raises(QuizGenerationError, match="exactly 4 options"):
        QuizGenerator(llm).generate(chunks, question_count=1)


def test_correct_label_must_match_an_option() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json(correct_label="E"))
    with pytest.raises(QuizGenerationError, match="correct_label"):
        QuizGenerator(llm).generate(chunks, question_count=1)


def test_unknown_source_citation_is_rejected() -> None:
    chunks = (_chunk("notes.pdf:p1:c1", "Stacks use LIFO."),)
    llm = FakeLLM(_quiz_json(source_citations=[2]))
    with pytest.raises(QuizGenerationError, match="unknown source"):
        QuizGenerator(llm).generate(chunks, question_count=1)
