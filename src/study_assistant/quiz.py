"""Generate multiple-choice quizzes from retrieved study chunks.

This module does not retrieve, embed, or call Ollama. Callers pass
`SimilarChunk` values and an `LLMClient`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from study_assistant.llm import LLMClient
from study_assistant.vector_store import SimilarChunk

DEFAULT_QUESTION_COUNT = 3
MAX_QUESTION_COUNT = 20
OPTION_LABELS = ("A", "B", "C", "D")
Difficulty = Literal["easy", "medium", "hard"]
DIFFICULTY_LEVELS: tuple[Difficulty, ...] = ("easy", "medium", "hard")
DEFAULT_DIFFICULTY: Difficulty = "medium"

_DIFFICULTY_INSTRUCTIONS: dict[Difficulty, str] = {
    "easy": "Write easy questions that test direct recall of facts stated in the context.",
    "medium": "Write medium-difficulty questions that require understanding the context.",
    "hard": "Write hard questions that require careful comparison or inference from the context without inventing facts.",
}


class QuizGenerationError(Exception):
    """Raised when quiz generation input or LLM output is invalid."""


@dataclass(frozen=True)
class AnswerOption:
    """One multiple-choice option."""

    label: str
    text: str


@dataclass(frozen=True)
class QuestionSource:
    """Trace from a question back to a retrieved chunk."""

    citation_index: int
    chunk_id: str
    source_path: Path
    page_number: int


@dataclass(frozen=True)
class QuizQuestion:
    """One grounded multiple-choice question."""

    question: str
    options: tuple[AnswerOption, ...]
    correct_label: str
    explanation: str
    sources: tuple[QuestionSource, ...]
    topic: str = ""


@dataclass(frozen=True)
class Quiz:
    """A set of multiple-choice questions."""

    questions: tuple[QuizQuestion, ...]
    difficulty: Difficulty = DEFAULT_DIFFICULTY
    quiz_id: str = field(default_factory=lambda: uuid4().hex)


class QuizGenerator:
    """Build a quiz from retrieved chunks using the existing LLM interface."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        chunks: Sequence[SimilarChunk],
        *,
        question_count: int = DEFAULT_QUESTION_COUNT,
        difficulty: Difficulty = DEFAULT_DIFFICULTY,
        topic_allocation: Sequence[tuple[str, int]] = (),
    ) -> Quiz:
        count = _validate_question_count(question_count)
        level = _validate_difficulty(difficulty)
        if not chunks:
            raise QuizGenerationError("Cannot generate a quiz without retrieved study material")

        prompt = build_quiz_prompt(
            chunks,
            question_count=count,
            difficulty=level,
            topic_allocation=topic_allocation,
        )
        raw = self.llm.complete(prompt)
        return parse_quiz_response(raw, chunks, question_count=count, difficulty=level)


def generate_quiz(
    chunks: Sequence[SimilarChunk],
    *,
    llm: LLMClient,
    question_count: int = DEFAULT_QUESTION_COUNT,
    difficulty: Difficulty = DEFAULT_DIFFICULTY,
    topic_allocation: Sequence[tuple[str, int]] = (),
) -> Quiz:
    """Convenience wrapper around `QuizGenerator.generate`."""
    return QuizGenerator(llm).generate(
        chunks,
        question_count=question_count,
        difficulty=difficulty,
        topic_allocation=topic_allocation,
    )


def build_quiz_prompt(
    chunks: Sequence[SimilarChunk],
    *,
    question_count: int,
    difficulty: Difficulty = DEFAULT_DIFFICULTY,
    topic_allocation: Sequence[tuple[str, int]] = (),
) -> str:
    """Deterministic quiz prompt. Includes only the supplied chunks."""
    difficulty = _validate_difficulty(difficulty)
    source_list = "\n".join(
        f"[{index}] {chunk.source_path.as_posix()}, page {chunk.page_number}, chunk_id={chunk.chunk_id}"
        for index, chunk in enumerate(chunks, start=1)
    )
    context_blocks = "\n\n".join(
        _format_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)
    )
    max_citation = len(chunks)
    topic_block = _format_topic_allocation(topic_allocation)
    return (
        "You are a study assistant that writes multiple-choice quizzes.\n"
        "Use ONLY the retrieved study context below.\n"
        "Do not use outside knowledge.\n"
        "Do not invent facts or unsupported claims.\n"
        "Do not invent sources or citation numbers.\n"
        "Every question must be answerable from the context.\n"
        "Write exactly one unambiguously correct option per question.\n"
        "Do not mark more than one option as correct.\n"
        "Do not write ambiguous questions where two options could be correct.\n"
        "Each question must have exactly 4 options labeled A, B, C, and D.\n"
        "Assign each question a short topic label drawn only from the context.\n"
        f"{topic_block}"
        f"Difficulty: {difficulty}.\n"
        f"{_DIFFICULTY_INSTRUCTIONS[difficulty]}\n"
        f"Generate exactly {question_count} question(s).\n"
        f"Cite sources with integers from 1 to {max_citation} matching the numbered context.\n"
        "Return JSON only, with this shape:\n"
        '{"questions":[{"question":"...","options":[{"label":"A","text":"..."},'
        '{"label":"B","text":"..."},{"label":"C","text":"..."},{"label":"D","text":"..."}],'
        '"correct_label":"A","explanation":"...","topic":"...","source_citations":[1]}]}\n'
        "\n"
        "Sources:\n"
        f"{source_list}\n"
        "\n"
        "Study context:\n"
        f"{context_blocks}\n"
    )


def parse_quiz_response(
    raw: str,
    chunks: Sequence[SimilarChunk],
    *,
    question_count: int,
    difficulty: Difficulty = DEFAULT_DIFFICULTY,
) -> Quiz:
    """Parse and validate LLM JSON into a `Quiz`."""
    level = _validate_difficulty(difficulty)
    payload = _load_json_object(raw)
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise QuizGenerationError("Quiz JSON must contain a 'questions' list")
    if len(raw_questions) != question_count:
        raise QuizGenerationError(
            f"Expected {question_count} question(s), got {len(raw_questions)}"
        )

    questions = tuple(_parse_question(item, chunks, index=index) for index, item in enumerate(raw_questions, start=1))
    return Quiz(questions=questions, difficulty=level)


def _format_topic_allocation(topic_allocation: Sequence[tuple[str, int]]) -> str:
    if not topic_allocation:
        return ""
    names: list[str] = []
    parts: list[str] = []
    for topic, count in topic_allocation:
        label = topic.strip()
        if not label:
            continue
        names.append(label)
        parts.append(f"{label}: {count}")
    if not names:
        return ""
    return (
        "Prioritize these topics (weakest first): "
        + ", ".join(names)
        + ".\n"
        "Question allocation: "
        + "; ".join(parts)
        + ".\n"
    )


def _validate_question_count(question_count: int) -> int:
    if question_count < 1:
        raise ValueError("question_count must be at least 1")
    if question_count > MAX_QUESTION_COUNT:
        raise ValueError(f"question_count must be at most {MAX_QUESTION_COUNT}")
    return question_count


def _validate_difficulty(difficulty: str) -> Difficulty:
    if isinstance(difficulty, str):
        normalized = difficulty.strip().lower()
        for level in DIFFICULTY_LEVELS:
            if normalized == level:
                return level
    raise ValueError("difficulty must be one of: easy, medium, hard")


def _format_chunk(index: int, chunk: SimilarChunk) -> str:
    body = chunk.text if chunk.text.strip() else "(no text extracted from this chunk)"
    return (
        f"[{index}] source={chunk.source_path.as_posix()} page={chunk.page_number} "
        f"chunk_id={chunk.chunk_id}\n{body}"
    )


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise QuizGenerationError("Quiz response is not valid JSON")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QuizGenerationError("Quiz response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise QuizGenerationError("Quiz JSON must be an object")
    return payload


def _parse_question(item: Any, chunks: Sequence[SimilarChunk], *, index: int) -> QuizQuestion:
    if not isinstance(item, dict):
        raise QuizGenerationError(f"Question {index} must be an object")

    question = _required_text(item.get("question"), field=f"Question {index} text")
    explanation = _required_text(item.get("explanation"), field=f"Question {index} explanation")
    options = _parse_options(item.get("options"), question_index=index)
    correct_label = str(item.get("correct_label", "")).strip().upper()
    labels = {option.label: option.text for option in options}
    if correct_label not in labels:
        raise QuizGenerationError(f"Question {index} correct_label must be one of A, B, C, D")

    sources = _parse_sources(item.get("source_citations"), chunks, question_index=index)
    topic = _optional_topic(item.get("topic"), question_index=index)
    return QuizQuestion(
        question=question,
        options=options,
        correct_label=correct_label,
        explanation=explanation,
        sources=sources,
        topic=topic,
    )


def _parse_options(raw_options: Any, *, question_index: int) -> tuple[AnswerOption, ...]:
    if not isinstance(raw_options, list) or len(raw_options) != 4:
        raise QuizGenerationError(f"Question {question_index} must have exactly 4 options")

    options: list[AnswerOption] = []
    for position, expected_label, raw_option in zip(range(4), OPTION_LABELS, raw_options, strict=True):
        if not isinstance(raw_option, dict):
            raise QuizGenerationError(f"Question {question_index} option {expected_label} must be an object")
        label = str(raw_option.get("label", "")).strip().upper()
        if label != expected_label:
            raise QuizGenerationError(
                f"Question {question_index} option {position + 1} must use label {expected_label}"
            )
        text = _required_text(raw_option.get("text"), field=f"Question {question_index} option {expected_label}")
        options.append(AnswerOption(label=label, text=text))

    texts = [option.text for option in options]
    if len(set(texts)) != 4:
        raise QuizGenerationError(f"Question {question_index} options must be unique")
    return tuple(options)


def _parse_sources(
    raw_citations: Any,
    chunks: Sequence[SimilarChunk],
    *,
    question_index: int,
) -> tuple[QuestionSource, ...]:
    if not isinstance(raw_citations, list) or not raw_citations:
        raise QuizGenerationError(f"Question {question_index} must cite at least one source")

    sources: list[QuestionSource] = []
    seen: set[int] = set()
    for raw_index in raw_citations:
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raise QuizGenerationError(f"Question {question_index} source_citations must be integers")
        if raw_index in seen:
            continue
        if raw_index < 1 or raw_index > len(chunks):
            raise QuizGenerationError(
                f"Question {question_index} cites unknown source [{raw_index}]"
            )
        chunk = chunks[raw_index - 1]
        sources.append(
            QuestionSource(
                citation_index=raw_index,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                page_number=chunk.page_number,
            )
        )
        seen.add(raw_index)
    return tuple(sources)


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuizGenerationError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_topic(value: Any, *, question_index: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise QuizGenerationError(f"Question {question_index} topic must be a string")
    return value.strip()
