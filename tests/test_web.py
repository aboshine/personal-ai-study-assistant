from __future__ import annotations

import io
from pathlib import Path

from datetime import datetime, timezone

from study_assistant.llm.errors import LLMConnectionError
from study_assistant.pdf_extraction import InvalidPdfError, PdfNotFoundError
from study_assistant.quiz import AnswerOption, QuestionSource, Quiz, QuizGenerationError, QuizQuestion
from study_assistant.quiz_attempts import QuizAttempt, create_quiz_attempt
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz
from study_assistant.rag import RagResult
from study_assistant.service import IndexedPdf
from study_assistant.study_planner import StudyPlannerItem, UnifiedStudyPlan
from study_assistant.vector_store import SimilarChunk
from tests.pdf_fixtures import minimal_pdf_bytes
from web.app import create_app


EXPLANATION = "Because a stack always removes the most recently added item."


class FakeAssistant:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        chunk_count: int = 4,
        ask_result: RagResult | None = None,
        ask_error: Exception | None = None,
        quiz: Quiz | None = None,
        quiz_error: Exception | None = None,
        attempts: tuple[QuizAttempt, ...] = (),
        plan: UnifiedStudyPlan | None = None,
    ) -> None:
        self.error = error
        self.chunk_count = chunk_count
        self.ask_result = ask_result
        self.ask_error = ask_error
        self.quiz = quiz if quiz is not None else _sample_quiz()
        self.quiz_error = quiz_error
        self.attempts = attempts
        self.plan = plan if plan is not None else UnifiedStudyPlan(items=())
        self.indexed_paths: list[Path] = []
        self.questions: list[str] = []
        self.quiz_calls: list[dict[str, object]] = []
        self.adaptive_calls: list[dict[str, object]] = []
        self.recorded: list[tuple[str, object]] = []

    def index_pdf(self, pdf_path: str | Path) -> IndexedPdf:
        path = Path(pdf_path)
        self.indexed_paths.append(path)
        if self.error is not None:
            raise self.error
        return IndexedPdf(source_path=path.resolve(), chunk_count=self.chunk_count)

    def ask(self, question: str) -> RagResult:
        self.questions.append(question)
        if self.ask_error is not None:
            raise self.ask_error
        if self.ask_result is not None:
            return self.ask_result
        return RagResult(answer="A stack uses LIFO. [1]", sources=())

    def generate_quiz(self, query: str, *, question_count: int = 3, difficulty: str = "medium") -> Quiz:
        self.quiz_calls.append(
            {"query": query, "question_count": question_count, "difficulty": difficulty}
        )
        if self.quiz_error is not None:
            raise self.quiz_error
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
        if self.quiz_error is not None:
            raise self.quiz_error
        return self.quiz

    def evaluate_and_record(self, quiz: Quiz, answers: object, *, timestamp=None):
        self.recorded.append((quiz.quiz_id, answers))
        evaluation = evaluate_quiz(quiz, answers)
        return evaluation, create_quiz_attempt(quiz, evaluation)

    def list_attempts(self) -> tuple[QuizAttempt, ...]:
        return self.attempts

    def current_study_plan(self, *, available_topics: tuple[str, ...] = (), as_of=None) -> UnifiedStudyPlan:
        return self.plan


def _sample_quiz() -> Quiz:
    return Quiz(
        quiz_id="web-quiz-1",
        difficulty="medium",
        questions=(
            QuizQuestion(
                question="What ordering does a stack use?",
                options=tuple(
                    AnswerOption(label=label, text=text)
                    for label, text in zip("ABCD", ("LIFO", "FIFO", "Heap", "Graph"), strict=True)
                ),
                correct_label="A",
                explanation=EXPLANATION,
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


def _client(tmp_path: Path, assistant: FakeAssistant):
    app = create_app(assistant=assistant, uploads_dir=tmp_path / "uploads")
    app.config["TESTING"] = True
    return app.test_client(), app


def test_library_page_has_nav_and_upload_form(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    response = client.get("/library")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'name="pdf"' in html
    assert ">Library</a>" in html
    assert ">Ask</a>" in html
    assert ">Quiz</a>" in html
    assert ">History</a>" in html
    assert ">Plan</a>" in html
    assert client.get("/").status_code == 302


def test_placeholder_pages_are_linked(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    for path, heading in (("/quiz", "Quiz"), ("/history", "History"), ("/plan", "Plan")):
        response = client.get(path)
        assert response.status_code == 200
        assert heading in response.get_data(as_text=True)


def _sample_rag_result() -> RagResult:
    return RagResult(
        answer="A stack uses LIFO ordering. [1]",
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


def test_ask_page_has_question_form(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    response = client.get("/ask")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "<h1>Ask</h1>" in html
    assert 'name="question"' in html
    assert ">Library</a>" in html


def test_ask_displays_answer_and_numbered_citations(tmp_path: Path) -> None:
    assistant = FakeAssistant(ask_result=_sample_rag_result())
    client, _app = _client(tmp_path, assistant)
    response = client.post("/ask", data={"question": "What is a stack?"})
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert assistant.questions == ["What is a stack?"]
    assert "A stack uses LIFO ordering. [1]" in html
    assert "[1] notes.pdf, page 1" in html
    assert "Traceback" not in html


def test_ask_empty_question_is_rejected(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, _app = _client(tmp_path, assistant)
    response = client.post("/ask", data={"question": "   "})
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Enter a question." in html
    assert assistant.questions == []


def test_ask_shows_rag_error_without_traceback(tmp_path: Path) -> None:
    assistant = FakeAssistant(
        ask_error=LLMConnectionError("Cannot connect to Ollama at http://127.0.0.1:11434. Is Ollama running?"),
    )
    client, _app = _client(tmp_path, assistant)
    response = client.post("/ask", data={"question": "What is a stack?"})
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Cannot connect to Ollama" in html
    assert "Traceback" not in html


def test_upload_saves_pdf_and_indexes_through_assistant(tmp_path: Path) -> None:
    assistant = FakeAssistant(chunk_count=3)
    client, app = _client(tmp_path, assistant)
    response = client.post(
        "/library",
        data={"pdf": (io.BytesIO(minimal_pdf_bytes(["Stacks use LIFO."])), "notes.pdf")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    html = response.get_data(as_text=True)
    saved = app.config["UPLOADS_DIR"] / "notes.pdf"
    assert response.status_code == 200
    assert saved.is_file()
    assert assistant.indexed_paths == [saved]
    assert "Indexed" in html
    assert "notes.pdf" in html
    assert "Stored chunks: 3" in html


def test_upload_without_file_shows_error(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, _app = _client(tmp_path, assistant)
    response = client.post("/library", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "Choose a PDF file to upload." in response.get_data(as_text=True)
    assert assistant.indexed_paths == []


def test_invalid_pdf_error_is_shown_without_traceback(tmp_path: Path) -> None:
    assistant = FakeAssistant(error=InvalidPdfError("Invalid or unreadable PDF: notes.pdf"))
    client, _app = _client(tmp_path, assistant)
    response = client.post(
        "/library",
        data={"pdf": (io.BytesIO(b"not-a-pdf"), "notes.pdf")},
        content_type="multipart/form-data",
    )
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Invalid or unreadable PDF: notes.pdf" in html
    assert "Traceback" not in html


def test_missing_pdf_error_is_shown_cleanly(tmp_path: Path) -> None:
    assistant = FakeAssistant(error=PdfNotFoundError("PDF not found: missing.pdf"))
    client, _app = _client(tmp_path, assistant)
    response = client.post(
        "/library",
        data={"pdf": (io.BytesIO(minimal_pdf_bytes(["text"])), "missing.pdf")},
        content_type="multipart/form-data",
    )
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "PDF not found: missing.pdf" in html
    assert "Traceback" not in html
    assert assistant.indexed_paths


def test_quiz_page_has_generate_form(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    response = client.get("/quiz")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'name="query"' in html
    assert 'name="count"' in html
    assert 'name="difficulty"' in html
    assert 'name="adaptive"' in html
    assert ">Library</a>" in html


def test_generate_quiz_keeps_answers_server_side(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, app = _client(tmp_path, assistant)
    response = client.post(
        "/quiz",
        data={"query": "stacks", "count": "2", "difficulty": "hard"},
        follow_redirects=True,
    )
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert assistant.quiz_calls == [{"query": "stacks", "question_count": 2, "difficulty": "hard"}]
    assert "web-quiz-1" in html
    assert "What ordering does a stack use?" in html
    assert "LIFO" in html
    assert EXPLANATION not in html
    assert "correct_label" not in html
    assert 'name="quiz_id"' in html
    assert "web-quiz-1" in app.config["PENDING_QUIZZES"]


def test_generate_adaptive_quiz_uses_history_flow(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, _app = _client(tmp_path, assistant)
    response = client.post(
        "/quiz",
        data={
            "query": "queues",
            "count": "1",
            "difficulty": "medium",
            "adaptive": "1",
            "topics": "Queues,Stacks",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert assistant.quiz_calls == []
    assert assistant.adaptive_calls == [
        {
            "query": "queues",
            "question_count": 1,
            "available_topics": ("Queues", "Stacks"),
            "difficulty": None,
        }
    ]
    assert "What ordering does a stack use?" in response.get_data(as_text=True)


def test_submit_quiz_records_attempt_and_shows_feedback(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, app = _client(tmp_path, assistant)
    client.post("/quiz", data={"query": "stacks", "count": "1", "difficulty": "medium"})
    response = client.post(
        "/quiz/submit",
        data={"quiz_id": "web-quiz-1", "answer_0": "A"},
    )
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert len(assistant.recorded) == 1
    assert assistant.recorded[0][0] == "web-quiz-1"
    assert "Score: 1/1" in html
    assert EXPLANATION in html
    assert "[1] notes.pdf, page 1" in html
    assert "correct" in html
    assert "web-quiz-1" not in app.config["PENDING_QUIZZES"]


def test_empty_quiz_query_is_rejected(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    client, _app = _client(tmp_path, assistant)
    response = client.post("/quiz", data={"query": "  ", "count": "3", "difficulty": "medium"})
    assert response.status_code == 200
    assert "Enter a quiz topic or query." in response.get_data(as_text=True)
    assert assistant.quiz_calls == []


def test_missing_quiz_state_is_handled_cleanly(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    take = client.get("/quiz/take/missing-id")
    assert take.status_code == 200
    assert "no longer available" in take.get_data(as_text=True)
    submit = client.post("/quiz/submit", data={"quiz_id": "missing-id", "answer_0": "A"})
    assert submit.status_code == 200
    assert "no longer available" in submit.get_data(as_text=True)


def test_quiz_generation_error_is_shown_without_traceback(tmp_path: Path) -> None:
    assistant = FakeAssistant(quiz_error=QuizGenerationError("Cannot generate a quiz without retrieved study material"))
    client, _app = _client(tmp_path, assistant)
    response = client.post("/quiz", data={"query": "stacks", "count": "3", "difficulty": "medium"})
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Cannot generate a quiz without retrieved study material" in html
    assert "Traceback" not in html


def test_history_empty_state(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    response = client.get("/history")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "<h1>History</h1>" in html
    assert "No quiz attempts yet." in html
    assert ">Quiz</a>" in html


def test_history_lists_persisted_attempts(tmp_path: Path) -> None:
    quiz = _sample_quiz()
    evaluation = evaluate_quiz(quiz, (SubmittedAnswer(0, "A"),))
    attempt = create_quiz_attempt(
        quiz,
        evaluation,
        timestamp=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
    )
    client, _app = _client(tmp_path, FakeAssistant(attempts=(attempt,)))
    response = client.get("/history")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "2026-09-12T12:00:00+00:00" in html
    assert "medium" in html
    assert "1/1" in html
    assert "Stacks" in html
    assert "No quiz attempts yet." not in html


def test_plan_empty_state(tmp_path: Path) -> None:
    client, _app = _client(tmp_path, FakeAssistant())
    response = client.get("/plan")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "<h1>Plan</h1>" in html
    assert "No study plan items." in html


def test_plan_shows_current_study_plan(tmp_path: Path) -> None:
    plan = UnifiedStudyPlan(
        items=(
            StudyPlannerItem(
                topic="Queues",
                priority=1,
                recommended_difficulty="easy",
                next_review=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
                reason="Low accuracy; practice at easy difficulty.",
            ),
        )
    )
    client, _app = _client(tmp_path, FakeAssistant(plan=plan))
    response = client.get("/plan")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Queues" in html
    assert "difficulty=easy" in html
    assert "2026-09-13T12:00:00+00:00" in html
    assert "Low accuracy; practice at easy difficulty." in html
    assert "No study plan items." not in html
