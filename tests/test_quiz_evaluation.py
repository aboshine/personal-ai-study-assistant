from pathlib import Path

import pytest

from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_evaluation import (
    QuizEvaluationError,
    SubmittedAnswer,
    evaluate_quiz,
)


def _options(*texts: str) -> tuple[AnswerOption, ...]:
    return tuple(AnswerOption(label=label, text=text) for label, text in zip("ABCD", texts, strict=True))


def _source(chunk_id: str, *, source: str = "notes.pdf", page: int = 1, citation: int = 1) -> QuestionSource:
    return QuestionSource(
        citation_index=citation,
        chunk_id=chunk_id,
        source_path=Path(source),
        page_number=page,
    )


def _question(
    text: str,
    correct_label: str,
    explanation: str,
    sources: tuple[QuestionSource, ...],
) -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=_options("LIFO", "FIFO", "Heap", "Graph"),
        correct_label=correct_label,
        explanation=explanation,
        sources=sources,
    )


def _two_question_quiz() -> Quiz:
    return Quiz(
        questions=(
            _question(
                "What ordering does a stack use?",
                "A",
                "Stacks use LIFO.",
                (_source("notes.pdf:p1:c1", page=1),),
            ),
            _question(
                "What ordering does a queue use?",
                "B",
                "Queues use FIFO.",
                (_source("ds.pdf:p3:c1", source="ds.pdf", page=3, citation=2),),
            ),
        )
    )


def test_all_answers_correct() -> None:
    result = evaluate_quiz(
        _two_question_quiz(),
        (
            SubmittedAnswer(0, "A"),
            SubmittedAnswer(1, "B"),
        ),
    )
    assert result.total_questions == 2
    assert result.correct_count == 2
    assert result.incorrect_count == 0
    assert result.unanswered_count == 0
    assert result.score_percent == 100.0
    assert all(item.is_correct for item in result.question_results)


def test_all_answers_incorrect() -> None:
    result = evaluate_quiz(
        _two_question_quiz(),
        (
            SubmittedAnswer(0, "C"),
            SubmittedAnswer(1, "A"),
        ),
    )
    assert result.correct_count == 0
    assert result.incorrect_count == 2
    assert result.unanswered_count == 0
    assert result.score_percent == 0.0
    assert all(not item.is_correct for item in result.question_results)


def test_mixed_and_unanswered_answers() -> None:
    result = evaluate_quiz(
        _two_question_quiz(),
        (SubmittedAnswer(0, "a"),),
    )
    assert result.correct_count == 1
    assert result.incorrect_count == 0
    assert result.unanswered_count == 1
    assert result.score_percent == 50.0
    first, second = result.question_results
    assert first.question_index == 0
    assert first.selected_label == "A"
    assert first.correct_label == "A"
    assert first.is_correct is True
    assert first.is_unanswered is False
    assert first.explanation == "Stacks use LIFO."
    assert first.sources[0].chunk_id == "notes.pdf:p1:c1"
    assert first.sources[0].page_number == 1
    assert first.topic == ""
    assert second.selected_label is None
    assert second.is_unanswered is True
    assert second.is_correct is False
    assert second.explanation == "Queues use FIFO."
    assert second.sources[0].source_path == Path("ds.pdf")
    assert second.sources[0].page_number == 3


def test_evaluation_copies_question_topic() -> None:
    quiz = Quiz(
        questions=(
            QuizQuestion(
                question="What ordering does a stack use?",
                options=_options("LIFO", "FIFO", "Heap", "Graph"),
                correct_label="A",
                explanation="Stacks use LIFO.",
                sources=(_source("notes.pdf:p1:c1", page=1),),
                topic="Stacks",
            ),
        )
    )
    result = evaluate_quiz(quiz, (SubmittedAnswer(0, "A"),))
    assert result.question_results[0].topic == "Stacks"


def test_explicit_unanswered_label() -> None:
    result = evaluate_quiz(
        _two_question_quiz(),
        (
            SubmittedAnswer(0, None),
            SubmittedAnswer(1, "B"),
        ),
    )
    assert result.unanswered_count == 1
    assert result.correct_count == 1
    assert result.incorrect_count == 0
    assert result.score_percent == 50.0
    assert result.question_results[0].is_unanswered is True


def test_invalid_option_label_is_rejected() -> None:
    with pytest.raises(QuizEvaluationError, match="Invalid option label"):
        evaluate_quiz(_two_question_quiz(), (SubmittedAnswer(0, "E"),))


def test_invalid_question_index_is_rejected() -> None:
    with pytest.raises(QuizEvaluationError, match="out of range"):
        evaluate_quiz(_two_question_quiz(), (SubmittedAnswer(2, "A"),))
    with pytest.raises(QuizEvaluationError, match="out of range"):
        evaluate_quiz(_two_question_quiz(), (SubmittedAnswer(-1, "A"),))


def test_duplicate_question_index_is_rejected() -> None:
    with pytest.raises(QuizEvaluationError, match="Duplicate answer"):
        evaluate_quiz(
            _two_question_quiz(),
            (SubmittedAnswer(0, "A"), SubmittedAnswer(0, "B")),
        )


def test_empty_quiz_scores_zero() -> None:
    result = evaluate_quiz(Quiz(questions=()), ())
    assert result.total_questions == 0
    assert result.correct_count == 0
    assert result.incorrect_count == 0
    assert result.unanswered_count == 0
    assert result.score_percent == 0.0
    assert result.question_results == ()


def test_empty_quiz_rejects_any_answer() -> None:
    with pytest.raises(QuizEvaluationError, match="out of range"):
        evaluate_quiz(Quiz(questions=()), (SubmittedAnswer(0, "A"),))


def test_evaluation_is_deterministic() -> None:
    quiz = _two_question_quiz()
    answers = (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "C"))
    first = evaluate_quiz(quiz, answers)
    second = evaluate_quiz(quiz, answers)
    assert first == second
    assert first.score_percent == 50.0
    assert first.question_results[1].is_correct is False
    assert first.question_results[1].correct_label == "B"
