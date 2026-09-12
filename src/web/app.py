"""Flask application factory. Maps HTTP to StudyAssistant; does not score or retrieve."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from study_assistant.config import Settings, load_settings
from study_assistant.embeddings.errors import EmbeddingError
from study_assistant.llm.errors import LLMError
from study_assistant.pdf_extraction import PdfExtractionError
from study_assistant.quiz import (
    DEFAULT_DIFFICULTY,
    DEFAULT_QUESTION_COUNT,
    DIFFICULTY_LEVELS,
    MAX_QUESTION_COUNT,
    OPTION_LABELS,
    Quiz,
    QuizGenerationError,
)
from study_assistant.quiz_attempt_store import QuizAttemptStoreError
from study_assistant.quiz_attempts import QuizAttempt, QuizAttemptError
from study_assistant.quiz_evaluation import QuizEvaluation, QuizEvaluationError, SubmittedAnswer
from study_assistant.rag import RagResult
from study_assistant.service import IndexedPdf, StudyAssistant, create_study_assistant
from study_assistant.vector_store import VectorStoreError

UPLOAD_EXTENSION = ".pdf"


def create_app(
    *,
    assistant: StudyAssistant | None = None,
    uploads_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> Flask:
    """Build one Flask app with one StudyAssistant instance."""
    if assistant is not None and uploads_dir is not None:
        service = assistant
        upload_root = Path(uploads_dir)
    else:
        resolved_settings = settings if settings is not None else load_settings()
        service = assistant if assistant is not None else create_study_assistant(resolved_settings)
        upload_root = Path(uploads_dir) if uploads_dir is not None else resolved_settings.data_dir / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-study-assistant")
    app.config["STUDY_ASSISTANT"] = service
    app.config["UPLOADS_DIR"] = upload_root.resolve()
    app.config["PENDING_QUIZZES"] = {}

    _register_routes(app)
    return app


def main() -> None:
    create_app().run(host="127.0.0.1", port=5000, debug=False)


def _register_routes(app: Flask) -> None:
    @app.get("/")
    def home():
        return redirect(url_for("library"))

    @app.route("/library", methods=["GET", "POST"])
    def library():
        indexed: IndexedPdf | None = None
        error: str | None = None
        if request.method == "POST":
            indexed, error = _handle_upload(app)
            if error is None and indexed is not None:
                flash(
                    f"Indexed {indexed.source_path.as_posix()}  Stored chunks: {indexed.chunk_count}"
                )
                return redirect(url_for("library"))
        return render_template("library.html", indexed=indexed, error=error, active="library")

    @app.route("/ask", methods=["GET", "POST"])
    def ask():
        result: RagResult | None = None
        error: str | None = None
        question = ""
        if request.method == "POST":
            question = (request.form.get("question") or "").strip()
            result, error = _handle_ask(app, question)
        return render_template(
            "ask.html",
            result=result,
            error=error,
            question=question,
            active="ask",
        )

    @app.route("/quiz", methods=["GET", "POST"])
    def quiz():
        error: str | None = None
        form = _quiz_form_defaults()
        if request.method == "POST":
            form = _quiz_form_from_request()
            generated, error = _handle_generate_quiz(app, form)
            if error is None and generated is not None:
                app.config["PENDING_QUIZZES"][generated.quiz_id] = generated
                return redirect(url_for("take_quiz", quiz_id=generated.quiz_id))
        return render_template(
            "quiz.html",
            error=error,
            form=form,
            difficulties=DIFFICULTY_LEVELS,
            question_counts=range(1, MAX_QUESTION_COUNT + 1),
            active="quiz",
        )

    @app.get("/quiz/take/<quiz_id>")
    def take_quiz(quiz_id: str):
        pending = app.config["PENDING_QUIZZES"].get(quiz_id)
        if pending is None:
            return render_template(
                "quiz.html",
                error="That quiz is no longer available. Generate a new one.",
                form=_quiz_form_defaults(),
                difficulties=DIFFICULTY_LEVELS,
                question_counts=range(1, MAX_QUESTION_COUNT + 1),
                active="quiz",
            )
        return render_template(
            "quiz_take.html",
            quiz_id=pending.quiz_id,
            difficulty=pending.difficulty,
            questions=_public_questions(pending),
            active="quiz",
        )

    @app.post("/quiz/submit")
    def submit_quiz():
        quiz_id = (request.form.get("quiz_id") or "").strip()
        pending = app.config["PENDING_QUIZZES"].get(quiz_id)
        if pending is None:
            return render_template(
                "quiz.html",
                error="That quiz is no longer available. Generate a new one.",
                form=_quiz_form_defaults(),
                difficulties=DIFFICULTY_LEVELS,
                question_counts=range(1, MAX_QUESTION_COUNT + 1),
                active="quiz",
            )
        evaluation, error = _handle_submit_quiz(app, pending)
        if error is not None:
            return render_template(
                "quiz_take.html",
                quiz_id=pending.quiz_id,
                difficulty=pending.difficulty,
                questions=_public_questions(pending),
                error=error,
                active="quiz",
            )
        app.config["PENDING_QUIZZES"].pop(quiz_id, None)
        return render_template(
            "quiz_results.html",
            quiz=pending,
            evaluation=evaluation,
            items=tuple(zip(pending.questions, evaluation.question_results, strict=True)),
            active="quiz",
        )

    @app.get("/history")
    def history():
        error: str | None = None
        attempts: tuple[QuizAttempt, ...] = ()
        try:
            attempts = app.config["STUDY_ASSISTANT"].list_attempts()
        except QuizAttemptStoreError as exc:
            error = str(exc)
        return render_template(
            "history.html",
            error=error,
            rows=_history_rows(attempts),
            active="history",
        )

    @app.get("/plan")
    def plan():
        study_plan = app.config["STUDY_ASSISTANT"].current_study_plan()
        return render_template(
            "plan.html",
            plan=study_plan,
            active="plan",
        )


def _handle_upload(app: Flask) -> tuple[IndexedPdf | None, str | None]:
    uploaded = request.files.get("pdf")
    if uploaded is None or not (uploaded.filename or "").strip():
        return None, "Choose a PDF file to upload."

    filename = secure_filename(Path(uploaded.filename).name)
    if not filename.lower().endswith(UPLOAD_EXTENSION):
        return None, "Upload a .pdf file."

    destination = app.config["UPLOADS_DIR"] / filename
    try:
        uploaded.save(destination)
        indexed = app.config["STUDY_ASSISTANT"].index_pdf(destination)
    except (PdfExtractionError, EmbeddingError, OSError, ValueError) as exc:
        return None, str(exc)
    return indexed, None


def _handle_ask(app: Flask, question: str) -> tuple[RagResult | None, str | None]:
    if not question:
        return None, "Enter a question."
    try:
        return app.config["STUDY_ASSISTANT"].ask(question), None
    except (EmbeddingError, LLMError, VectorStoreError, ValueError, OSError) as exc:
        return None, str(exc)


def _history_rows(attempts: tuple[QuizAttempt, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for attempt in reversed(attempts):
        topics: list[str] = []
        for result in attempt.question_results:
            topic = result.topic.strip()
            if topic and topic not in topics:
                topics.append(topic)
        rows.append({"attempt": attempt, "topics": tuple(topics)})
    return tuple(rows)


def _quiz_form_defaults() -> dict[str, str]:
    return {
        "query": "",
        "count": str(DEFAULT_QUESTION_COUNT),
        "difficulty": DEFAULT_DIFFICULTY,
        "adaptive": "",
        "topics": "",
    }


def _quiz_form_from_request() -> dict[str, str]:
    return {
        "query": (request.form.get("query") or "").strip(),
        "count": (request.form.get("count") or str(DEFAULT_QUESTION_COUNT)).strip(),
        "difficulty": (request.form.get("difficulty") or DEFAULT_DIFFICULTY).strip(),
        "adaptive": (request.form.get("adaptive") or "").strip(),
        "topics": (request.form.get("topics") or "").strip(),
    }


def _public_questions(quiz: Quiz) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "index": index,
            "prompt": question.question,
            "topic": question.topic,
            "options": question.options,
        }
        for index, question in enumerate(quiz.questions)
    )


def _handle_generate_quiz(app: Flask, form: dict[str, str]) -> tuple[Quiz | None, str | None]:
    query = form["query"]
    if not query:
        return None, "Enter a quiz topic or query."
    try:
        count = int(form["count"])
    except ValueError:
        return None, "Question count must be a number."
    difficulty = form["difficulty"]
    if difficulty not in DIFFICULTY_LEVELS:
        return None, "Difficulty must be easy, medium, or hard."
    topics = tuple(part.strip() for part in form["topics"].split(",") if part.strip())
    assistant: StudyAssistant = app.config["STUDY_ASSISTANT"]
    try:
        if form["adaptive"]:
            quiz = assistant.generate_adaptive_quiz(
                query,
                question_count=count,
                available_topics=topics,
            )
        else:
            quiz = assistant.generate_quiz(
                query,
                question_count=count,
                difficulty=difficulty,  # type: ignore[arg-type]
            )
    except (QuizGenerationError, EmbeddingError, LLMError, VectorStoreError, ValueError, OSError) as exc:
        return None, str(exc)
    return quiz, None


def _handle_submit_quiz(
    app: Flask,
    quiz: Quiz,
) -> tuple[QuizEvaluation | None, str | None]:
    answers = tuple(
        SubmittedAnswer(index, _submitted_label(request.form.get(f"answer_{index}")))
        for index in range(len(quiz.questions))
    )
    try:
        evaluation, _attempt = app.config["STUDY_ASSISTANT"].evaluate_and_record(quiz, answers)
    except (QuizEvaluationError, QuizAttemptError, ValueError) as exc:
        return None, str(exc)
    return evaluation, None


def _submitted_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip().upper()
    if not cleaned or cleaned == "-":
        return None
    if cleaned not in OPTION_LABELS:
        return cleaned
    return cleaned
