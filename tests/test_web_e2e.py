"""Web MVP smoke: Library → Ask → Quiz → History → Adaptive → Plan."""

from __future__ import annotations

import io
import re
from pathlib import Path

from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore
from study_assistant.rag import NO_CONTEXT_ANSWER, RagPipeline
from study_assistant.service import StudyAssistant
from study_assistant.vector_store import SqliteVectorStore
from tests.pdf_fixtures import minimal_pdf_bytes
from tests.test_e2e_smoke import KeywordEmbeddings, PipelineLLM
from web.app import create_app


def _web_assistant(tmp_path: Path) -> tuple[StudyAssistant, PipelineLLM, SqliteVectorStore, SqliteQuizAttemptStore]:
    vector_store = SqliteVectorStore(tmp_path / "vectors.sqlite")
    attempt_store = SqliteQuizAttemptStore(tmp_path / "attempts.sqlite")
    llm = PipelineLLM()
    assistant = StudyAssistant(
        rag=RagPipeline(
            embeddings=KeywordEmbeddings(),
            store=vector_store,
            llm=llm,
            top_k=2,
        ),
        attempt_store=attempt_store,
    )
    return assistant, llm, vector_store, attempt_store


def _client(tmp_path: Path, assistant: StudyAssistant):
    app = create_app(assistant=assistant, uploads_dir=tmp_path / "uploads")
    app.config["TESTING"] = True
    return app.test_client(), app


def _nav_ok(html: str) -> None:
    assert ">Library</a>" in html
    assert ">Ask</a>" in html
    assert ">Quiz</a>" in html
    assert ">History</a>" in html
    assert ">Plan</a>" in html


def test_web_mvp_workflow(tmp_path: Path) -> None:
    assistant, llm, vector_store, attempt_store = _web_assistant(tmp_path)
    client, app = _client(tmp_path, assistant)
    try:
        for path in ("/library", "/ask", "/quiz", "/history", "/plan"):
            response = client.get(path)
            assert response.status_code == 200
            _nav_ok(response.get_data(as_text=True))
        assert client.get("/").status_code == 302

        history = client.get("/history").get_data(as_text=True)
        assert "No quiz attempts yet." in history
        plan = client.get("/plan").get_data(as_text=True)
        assert "No study plan items." in plan

        empty_ask = client.post("/ask", data={"question": "What is a stack?"})
        assert empty_ask.status_code == 200
        assert NO_CONTEXT_ANSWER in empty_ask.get_data(as_text=True)

        empty_quiz = client.post(
            "/quiz",
            data={"query": "stack and queue", "count": "2", "difficulty": "medium"},
        )
        assert empty_quiz.status_code == 200
        assert "Cannot generate a quiz without retrieved study material" in empty_quiz.get_data(as_text=True)

        pdf_bytes = minimal_pdf_bytes(
            [
                "A stack uses LIFO ordering.",
                "A queue uses FIFO ordering.",
            ]
        )
        uploaded = client.post(
            "/library",
            data={"pdf": (io.BytesIO(pdf_bytes), "notes.pdf")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        uploaded_html = uploaded.get_data(as_text=True)
        assert uploaded.status_code == 200
        assert "Indexed" in uploaded_html
        assert "notes.pdf" in uploaded_html
        assert "Stored chunks:" in uploaded_html
        assert vector_store.count() >= 1

        asked = client.post("/ask", data={"question": "What is a stack?"})
        asked_html = asked.get_data(as_text=True)
        assert asked.status_code == 200
        assert "LIFO" in asked_html
        assert "[1]" in asked_html
        assert "notes.pdf" in asked_html
        assert "page 1" in asked_html

        take = client.post(
            "/quiz",
            data={"query": "stack and queue", "count": "2", "difficulty": "medium"},
            follow_redirects=True,
        )
        take_html = take.get_data(as_text=True)
        assert take.status_code == 200
        assert "What is true of Stacks?" in take_html
        assert "What is true of Queues?" in take_html
        assert "Stacks fact." not in take_html
        assert "Queues fact." not in take_html
        assert "correct_label" not in take_html
        quiz_id_match = re.search(r'name="quiz_id" value="([^"]+)"', take_html)
        assert quiz_id_match is not None
        quiz_id = quiz_id_match.group(1)

        results = client.post(
            "/quiz/submit",
            data={"quiz_id": quiz_id, "answer_0": "A", "answer_1": "A"},
        )
        results_html = results.get_data(as_text=True)
        assert results.status_code == 200
        assert "Score: 1/2" in results_html
        assert "Stacks fact." in results_html
        assert "notes.pdf" in results_html
        assert len(attempt_store.list_all()) == 1
        assert assistant.list_attempts()[0].evaluation.correct_count == 1

        history_after = client.get("/history").get_data(as_text=True)
        assert "No quiz attempts yet." not in history_after
        assert "1/2" in history_after
        assert "Stacks" in history_after
        assert "Queues" in history_after

        adaptive = client.post(
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
        adaptive_html = adaptive.get_data(as_text=True)
        assert adaptive.status_code == 200
        assert "Difficulty: easy" in adaptive_html
        assert "What is true of Queues?" in adaptive_html
        assert "Queues fact." not in adaptive_html
        assert "Prioritize these topics (weakest first): Queues" in llm.prompts[-1]

        plan_after = client.get("/plan").get_data(as_text=True)
        assert "No study plan items." not in plan_after
        assert plan_after.index("Queues") < plan_after.index("Stacks")
        assert "difficulty=easy" in plan_after
        assert "review=" in plan_after
    finally:
        vector_store.close()
        attempt_store.close()
