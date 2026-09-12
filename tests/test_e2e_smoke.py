"""End-to-end smoke test: PDF → index → RAG → quiz → attempt → adaptive → plan."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from study_assistant.chunking import chunk_document
from study_assistant.cli import main
from study_assistant.embeddings import EmbeddingClient, embed_chunks
from study_assistant.llm.base import LLMClient
from study_assistant.pdf_extraction import extract_pdf
from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore
from study_assistant.quiz_evaluation import SubmittedAnswer
from study_assistant.rag import RagPipeline
from study_assistant.service import StudyAssistant
from study_assistant.topic_tracking import summarize_topic_performance
from study_assistant.vector_store import SqliteVectorStore
from tests.pdf_fixtures import minimal_pdf_bytes

AS_OF = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class KeywordEmbeddings(EmbeddingClient):
    """Deterministic fake embeddings. Stack text aligns with stack queries."""

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        stack = 1.0 if "stack" in lowered or "lifo" in lowered else 0.0
        queue = 1.0 if "queue" in lowered or "fifo" in lowered else 0.0
        if stack == 0.0 and queue == 0.0:
            return [0.2, 0.2]
        return [stack, queue]


class PipelineLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "multiple-choice quizzes" in prompt:
            if "Generate exactly 1 question(s)." in prompt:
                return _quiz_payload(1, topic="Queues")
            return _quiz_payload(2)
        return "A stack uses LIFO ordering. [1]"


def _quiz_payload(count: int, *, topic: str = "Stacks") -> str:
    topics = ("Stacks", "Queues") if count == 2 else (topic,)
    questions = [
        {
            "question": f"What is true of {name}?",
            "options": [
                {"label": "A", "text": f"{name}-A-{index}"},
                {"label": "B", "text": f"{name}-B-{index}"},
                {"label": "C", "text": f"{name}-C-{index}"},
                {"label": "D", "text": f"{name}-D-{index}"},
            ],
            "correct_label": "A" if name == "Stacks" else "B",
            "explanation": f"{name} fact.",
            "topic": name,
            "source_citations": [1],
        }
        for index, name in enumerate(topics)
    ]
    return json.dumps({"questions": questions})


def test_pdf_through_study_plan_smoke(tmp_path: Path) -> None:
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(
        minimal_pdf_bytes(
            [
                "A stack uses LIFO ordering.",
                "A queue uses FIFO ordering.",
            ]
        )
    )

    document = extract_pdf(pdf_path)
    chunks = chunk_document(document, chunk_size=200, overlap=0)
    assert chunks
    assert any("stack" in chunk.text.lower() or "lifo" in chunk.text.lower() for chunk in chunks)

    embeddings = KeywordEmbeddings()
    embedded = embed_chunks(chunks, client=embeddings)
    assert len(embedded) == len(chunks)
    assert all(chunk.vector for chunk in embedded)

    vector_store = SqliteVectorStore(tmp_path / "vectors.sqlite")
    attempt_store = SqliteQuizAttemptStore(tmp_path / "attempts.sqlite")
    llm = PipelineLLM()
    try:
        vector_store.upsert(embedded)
        assert vector_store.count() == len(embedded)

        assistant = StudyAssistant(
            rag=RagPipeline(embeddings=embeddings, store=vector_store, llm=llm, top_k=2),
            attempt_store=attempt_store,
        )

        rag = assistant.ask("What is a stack?")
        assert "LIFO" in rag.answer
        assert rag.sources
        assert rag.sources[0].page_number == 1

        quiz = assistant.generate_quiz("stack and queue", question_count=2)
        assert len(quiz.questions) == 2
        assert {question.topic for question in quiz.questions} == {"Stacks", "Queues"}
        assert quiz.questions[0].sources[0].page_number >= 1

        evaluation = assistant.evaluate_quiz(
            quiz,
            (SubmittedAnswer(0, "A"), SubmittedAnswer(1, "A")),
        )
        assert evaluation.total_questions == 2
        assert evaluation.correct_count == 1
        assert evaluation.incorrect_count == 1

        attempt = assistant.record_attempt(quiz, evaluation, timestamp=AS_OF)
        loaded = attempt_store.get(attempt.attempt_id)
        assert loaded == attempt

        stats = summarize_topic_performance(attempt_store.list_all())
        by_topic = {item.topic: item for item in stats}
        assert by_topic["Stacks"].accuracy == 100.0
        assert by_topic["Queues"].accuracy == 0.0

        adaptive = assistant.generate_adaptive_quiz(
            "queues",
            question_count=1,
            available_topics=("Stacks", "Queues"),
        )
        assert adaptive.difficulty == "easy"
        assert "Prioritize these topics (weakest first): Queues" in llm.prompts[-1]

        plan = assistant.current_study_plan(as_of=AS_OF)
        assert plan.items[0].topic == "Queues"
        assert plan.items[0].recommended_difficulty == "easy"
        assert plan.items[1].topic == "Stacks"
    finally:
        vector_store.close()
        attempt_store.close()


def test_cli_mvp_workflow(tmp_path: Path, capsys) -> None:
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(
        minimal_pdf_bytes(
            [
                "A stack uses LIFO ordering.",
                "A queue uses FIFO ordering.",
            ]
        )
    )
    quiz_file = tmp_path / "quiz.json"
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
    try:
        assert main(["index", str(pdf_path)], assistant=assistant) == 0
        indexed = capsys.readouterr()
        assert "Indexed" in indexed.out
        assert "Stored chunks:" in indexed.out
        assert vector_store.count() >= 1

        assert main(["ask", "What is a stack?"], assistant=assistant) == 0
        asked = capsys.readouterr()
        assert "LIFO" in asked.out
        assert "page" in asked.out

        assert main(["quiz", "stack and queue", "--count", "2"], assistant=assistant) == 0
        quiz_out = capsys.readouterr().out
        assert "Quiz JSON:" in quiz_out
        quiz_json = quiz_out.split("Quiz JSON:\n", 1)[1]
        quiz_file.write_text(quiz_json, encoding="utf-8")
        payload = json.loads(quiz_json)
        assert len(payload["questions"]) == 2

        assert main(["evaluate", str(quiz_file), "A", "A"], assistant=assistant) == 0
        evaluated = capsys.readouterr()
        assert "Score: 1/2" in evaluated.out
        attempts = attempt_store.list_all()
        assert len(attempts) == 1
        assert attempts[0].evaluation.correct_count == 1
        assert attempts[0].evaluation.incorrect_count == 1

        assert main(
            ["adaptive", "queues", "--count", "1", "--topics", "Queues,Stacks"],
            assistant=assistant,
        ) == 0
        adaptive = capsys.readouterr()
        assert "difficulty=easy" in adaptive.out
        assert "Queues" in adaptive.out
        assert "Prioritize these topics (weakest first): Queues" in llm.prompts[-1]

        assert main(["plan"], assistant=assistant) == 0
        plan = capsys.readouterr()
        assert "Study plan:" in plan.out
        assert "Queues" in plan.out
        queues_pos = plan.out.index("Queues")
        stacks_pos = plan.out.index("Stacks")
        assert queues_pos < stacks_pos
    finally:
        vector_store.close()
        attempt_store.close()
