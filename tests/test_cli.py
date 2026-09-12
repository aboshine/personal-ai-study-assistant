from datetime import datetime, timezone
from pathlib import Path

from study_assistant.cli import _quiz_to_dict, main
from study_assistant.llm.errors import LLMConnectionError
from study_assistant.pdf_extraction import InvalidPdfError, PdfNotFoundError
from study_assistant.service import IndexedPdf
from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizQuestion
from study_assistant.quiz_evaluation import QuizEvaluation, QuestionResult, QuizEvaluationError
from study_assistant.rag import RagResult
from study_assistant.study_planner import StudyPlannerItem, UnifiedStudyPlan
from study_assistant.vector_store import SimilarChunk


def test_ask_pdf_prints_answer(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "study_assistant.cli.answer_from_pdf",
        lambda pdf_path, question: f"{pdf_path}:{question}:Stacks use LIFO.",
    )

    code = main(["ask-pdf", "notes.pdf", "What", "is", "a", "stack?"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "Answer:\nnotes.pdf:What is a stack?:Stacks use LIFO.\n"
    assert captured.err == ""


def test_ask_pdf_requires_path_and_question(capsys) -> None:
    code = main(["ask-pdf", "notes.pdf"])

    captured = capsys.readouterr()
    assert code == 2
    assert "PDF path and a question" in captured.err
    assert captured.out == ""


def test_ask_pdf_prints_service_error_without_traceback(capsys, monkeypatch) -> None:
    def fake_answer(pdf_path: str, question: str) -> str:
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")

    monkeypatch.setattr("study_assistant.cli.answer_from_pdf", fake_answer)

    code = main(["ask-pdf", "missing.pdf", "What is a stack?"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: PDF not found: missing.pdf" in captured.err
    assert "Traceback" not in captured.err


def test_ask_pdf_prints_llm_error_without_traceback(capsys, monkeypatch) -> None:
    def fake_answer(pdf_path: str, question: str) -> str:
        raise LLMConnectionError("Cannot connect to Ollama at http://127.0.0.1:11434. Is Ollama running?")

    monkeypatch.setattr("study_assistant.cli.answer_from_pdf", fake_answer)

    code = main(["ask-pdf", "notes.pdf", "What is a stack?"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Error: Cannot connect to Ollama" in captured.err
    assert "Traceback" not in captured.err


def test_help_prints_ask_pdf_usage(capsys) -> None:
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "ask-pdf PDF_PATH QUESTION" in captured.out
    assert "ask QUESTION" in captured.out
    assert "quiz QUERY" in captured.out
    assert "index PDF_PATH" in captured.out


class FakeAssistant:
    def __init__(self) -> None:
        self.quiz = _sample_quiz()
        self.asks: list[str] = []
        self.indexed: list[str] = []
        self.recorded: list[tuple[Quiz, QuizEvaluation]] = []
        self.quiz_calls: list[dict[str, object]] = []
        self.adaptive_calls: list[dict[str, object]] = []

    def index_pdf(self, pdf_path: str) -> IndexedPdf:
        self.indexed.append(pdf_path)
        return IndexedPdf(source_path=Path(pdf_path), chunk_count=2)

    def ask(self, question: str) -> RagResult:
        self.asks.append(question)
        return RagResult(
            answer="A stack uses LIFO.",
            sources=(
                SimilarChunk(
                    chunk_id="notes.pdf:p1:c1",
                    source_path=Path("notes.pdf"),
                    page_number=1,
                    text="Stacks use LIFO.",
                    similarity=0.9,
                ),
            ),
        )

    def generate_quiz(self, query: str, *, question_count: int = 3, difficulty: str = "medium") -> Quiz:
        self.quiz_calls.append({"query": query, "question_count": question_count, "difficulty": difficulty})
        return self.quiz

    def generate_adaptive_quiz(
        self,
        query: str,
        *,
        question_count: int = 3,
        available_topics: tuple[str, ...] = (),
        difficulty: str | None = None,
    ) -> Quiz:
        self.adaptive_calls.append(
            {
                "query": query,
                "question_count": question_count,
                "available_topics": available_topics,
                "difficulty": difficulty,
            }
        )
        return self.quiz

    def evaluate_quiz(self, quiz: Quiz, answers: object) -> QuizEvaluation:
        first = quiz.questions[0]
        selected = answers[0].selected_label if answers else None
        is_correct = selected == first.correct_label
        return QuizEvaluation(
            total_questions=1,
            correct_count=1 if is_correct else 0,
            incorrect_count=0 if is_correct or selected is None else 1,
            unanswered_count=1 if selected is None else 0,
            score_percent=100.0 if is_correct else 0.0,
            question_results=(
                QuestionResult(
                    question_index=0,
                    selected_label=selected,
                    correct_label=first.correct_label,
                    is_correct=is_correct,
                    is_unanswered=selected is None,
                    explanation=first.explanation,
                    sources=first.sources,
                    topic=first.topic,
                ),
            ),
        )

    def evaluate_and_record(self, quiz: Quiz, answers: object, *, timestamp=None) -> tuple[QuizEvaluation, object]:
        evaluation = self.evaluate_quiz(quiz, answers)
        self.recorded.append((quiz, evaluation))
        return evaluation, object()

    def current_study_plan(self, *, available_topics: tuple[str, ...] = (), as_of=None) -> UnifiedStudyPlan:
        return UnifiedStudyPlan(
            items=(
                StudyPlannerItem(
                    topic="Queues",
                    priority=1,
                    recommended_difficulty="easy",
                    next_review=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
                    reason="Low accuracy; practice at easy difficulty.",
                ),
            )
        )


def _sample_quiz() -> Quiz:
    return Quiz(
        quiz_id="quiz-cli",
        difficulty="medium",
        questions=(
            QuizQuestion(
                question="What ordering does a stack use?",
                options=tuple(
                    AnswerOption(label=label, text=text)
                    for label, text in zip("ABCD", ("LIFO", "FIFO", "Heap", "Graph"), strict=True)
                ),
                correct_label="A",
                explanation="Stacks use LIFO.",
                sources=(
                    QuestionSource(
                        citation_index=1,
                        chunk_id="notes.pdf:p1:c1",
                        source_path=Path("notes.pdf"),
                        page_number=1,
                    ),
                ),
                topic="Stacks",
            ),
        ),
    )


def test_index_command_prints_path_and_chunk_count(capsys) -> None:
    assistant = FakeAssistant()
    code = main(["index", "notes.pdf"], assistant=assistant)
    captured = capsys.readouterr()
    assert code == 0
    assert assistant.indexed == ["notes.pdf"]
    assert "Indexed notes.pdf" in captured.out
    assert "Stored chunks: 2" in captured.out
    assert captured.err == ""


def test_index_requires_a_pdf_path(capsys) -> None:
    code = main(["index"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 2
    assert "index requires a PDF path" in captured.err


def test_index_prints_invalid_pdf_without_traceback(capsys) -> None:
    class InvalidPdfAssistant:
        def index_pdf(self, pdf_path: str) -> IndexedPdf:
            raise InvalidPdfError(f"Invalid or unreadable PDF: {pdf_path}")

    code = main(["index", "notes.pdf"], assistant=InvalidPdfAssistant())
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: Invalid or unreadable PDF: notes.pdf" in captured.err
    assert "Traceback" not in captured.err
    class MissingPdfAssistant:
        def index_pdf(self, pdf_path: str) -> IndexedPdf:
            raise PdfNotFoundError(f"PDF not found: {pdf_path}")

    code = main(["index", "missing.pdf"], assistant=MissingPdfAssistant())
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: PDF not found: missing.pdf" in captured.err
    assert "Traceback" not in captured.err


def test_ask_command_uses_study_assistant(capsys) -> None:
    assistant = FakeAssistant()
    code = main(["ask", "What", "is", "a", "stack?"], assistant=assistant)
    captured = capsys.readouterr()
    assert code == 0
    assert assistant.asks == ["What is a stack?"]
    assert "A stack uses LIFO." in captured.out
    assert "notes.pdf, page 1" in captured.out


def test_ask_requires_question(capsys) -> None:
    code = main(["ask"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 2
    assert "ask requires a question" in captured.err


def test_quiz_and_adaptive_commands(capsys) -> None:
    assistant = FakeAssistant()
    code = main(["quiz", "stacks", "--count", "2", "--difficulty", "hard"], assistant=assistant)
    captured = capsys.readouterr()
    assert code == 0
    assert assistant.quiz_calls == [{"query": "stacks", "question_count": 2, "difficulty": "hard"}]
    assert "What ordering does a stack use?" in captured.out
    assert "Quiz JSON:" in captured.out

    code = main(["adaptive", "ds", "--topics", "Queues,Stacks"], assistant=assistant)
    captured = capsys.readouterr()
    assert code == 0
    assert assistant.adaptive_calls[0]["available_topics"] == ("Queues", "Stacks")
    assert "difficulty=medium" in captured.out


def test_plan_command(capsys) -> None:
    code = main(["plan"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 0
    assert "1. Queues" in captured.out
    assert "difficulty=easy" in captured.out


def test_evaluate_command(tmp_path: Path, capsys) -> None:
    import json

    quiz_file = tmp_path / "quiz.json"
    quiz = _sample_quiz()
    quiz_file.write_text(json.dumps(_quiz_to_dict(quiz)), encoding="utf-8")
    assistant = FakeAssistant()
    code = main(["evaluate", str(quiz_file), "A"], assistant=assistant)
    captured = capsys.readouterr()
    assert code == 0
    assert "Score: 1/1" in captured.out
    assert "correct" in captured.out
    assert len(assistant.recorded) == 1
    recorded_quiz, recorded_eval = assistant.recorded[0]
    assert recorded_quiz.quiz_id == quiz.quiz_id
    assert recorded_eval.correct_count == 1


def test_evaluate_rejects_invalid_quiz_json(tmp_path: Path, capsys) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    code = main(["evaluate", str(bad_json), "A"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: invalid quiz JSON:" in captured.err
    assert "Traceback" not in captured.err

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    code = main(["evaluate", str(not_object), "A"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 2
    assert "Quiz JSON must be an object" in captured.err
    assert "Traceback" not in captured.err


def test_evaluate_prints_invalid_answer_without_traceback(tmp_path: Path, capsys) -> None:
    import json

    class InvalidAnswerAssistant:
        def evaluate_and_record(self, quiz, answers, *, timestamp=None):
            raise QuizEvaluationError("Answer label must be one of: A, B, C, D")

    quiz_file = tmp_path / "quiz.json"
    quiz_file.write_text(json.dumps(_quiz_to_dict(_sample_quiz())), encoding="utf-8")
    code = main(["evaluate", str(quiz_file), "E"], assistant=InvalidAnswerAssistant())
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: Answer label must be one of: A, B, C, D" in captured.err
    assert "Traceback" not in captured.err


def test_evaluate_and_quiz_reject_missing_input(capsys) -> None:
    assert main(["quiz"], assistant=FakeAssistant()) == 2
    assert main(["adaptive"], assistant=FakeAssistant()) == 2
    assert main(["evaluate"], assistant=FakeAssistant()) == 2
    assert "quiz requires a query" in capsys.readouterr().err
    assert main(["evaluate", "missing.json", "A"], assistant=FakeAssistant()) == 2
    assert "quiz file not found" in capsys.readouterr().err


def test_unknown_option_is_rejected(capsys) -> None:
    code = main(["quiz", "stacks", "--nope"], assistant=FakeAssistant())
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown option" in captured.err

