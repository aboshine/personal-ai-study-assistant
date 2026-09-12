"""Thin Study Assistant service: RAG, quizzes, attempts, and study planning.

This module does not implement retrieval, scoring, or planning algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from study_assistant.adaptive_quiz import generate_adaptive_quiz
from study_assistant.chunking import chunk_document
from study_assistant.config import Settings
from study_assistant.embeddings import embed_chunks
from study_assistant.pdf_extraction import extract_pdf
from study_assistant.quiz import (
    DEFAULT_DIFFICULTY,
    DEFAULT_QUESTION_COUNT,
    Difficulty,
    Quiz,
    QuizGenerator,
)
from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore, get_quiz_attempt_store
from study_assistant.quiz_attempts import QuizAttempt, create_quiz_attempt
from study_assistant.quiz_evaluation import QuizEvaluation, SubmittedAnswer, evaluate_quiz
from study_assistant.rag import RagPipeline, RagResult, create_rag_pipeline
from study_assistant.study_planner import StudyPlanner, UnifiedStudyPlan
from study_assistant.vector_store import SimilarChunk


@dataclass(frozen=True)
class IndexedPdf:
    """Result of indexing one PDF into the existing vector store."""

    source_path: Path
    chunk_count: int


class StudyAssistant:
    """Orchestrate existing study-assistant modules."""

    def __init__(
        self,
        *,
        rag: RagPipeline,
        attempt_store: SqliteQuizAttemptStore,
        planner: StudyPlanner | None = None,
        quiz_generator: QuizGenerator | None = None,
    ) -> None:
        self.rag = rag
        self.attempt_store = attempt_store
        self.planner = planner if planner is not None else StudyPlanner()
        self.quiz_generator = quiz_generator if quiz_generator is not None else QuizGenerator(rag.llm)

    def ask(self, question: str) -> RagResult:
        """Answer a question using the existing RAG pipeline."""
        return self.rag.answer(question)

    def index_pdf(self, pdf_path: str | Path) -> IndexedPdf:
        """Extract, chunk, embed, and store one PDF, replacing any prior chunks for it."""
        document = extract_pdf(pdf_path)
        chunks = chunk_document(document)
        embedded = embed_chunks(chunks, client=self.rag.embeddings)
        self.rag.store.delete_by_source(document.source_path)
        if embedded:
            self.rag.store.upsert(embedded)
        return IndexedPdf(source_path=document.source_path, chunk_count=len(embedded))

    def generate_quiz(
        self,
        query: str,
        *,
        question_count: int = DEFAULT_QUESTION_COUNT,
        difficulty: Difficulty = DEFAULT_DIFFICULTY,
    ) -> Quiz:
        """Retrieve study chunks for `query`, then generate a quiz from them."""
        chunks = self._retrieve(query)
        return self.quiz_generator.generate(
            chunks,
            question_count=question_count,
            difficulty=difficulty,
        )

    def evaluate_quiz(self, quiz: Quiz, answers: Sequence[SubmittedAnswer]) -> QuizEvaluation:
        """Score submitted answers with the existing evaluator."""
        return evaluate_quiz(quiz, answers)

    def evaluate_and_record(
        self,
        quiz: Quiz,
        answers: Sequence[SubmittedAnswer],
        *,
        timestamp: datetime | None = None,
    ) -> tuple[QuizEvaluation, QuizAttempt]:
        """Score a quiz and persist the attempt with existing evaluation and store logic."""
        evaluation = self.evaluate_quiz(quiz, answers)
        attempt = self.record_attempt(quiz, evaluation, timestamp=timestamp)
        return evaluation, attempt

    def record_attempt(
        self,
        quiz: Quiz,
        evaluation: QuizEvaluation,
        *,
        timestamp: datetime | None = None,
    ) -> QuizAttempt:
        """Create an attempt and persist it in the attempt store."""
        attempt = create_quiz_attempt(quiz, evaluation, timestamp=timestamp)
        self.attempt_store.add(attempt)
        return attempt

    def list_attempts(self) -> tuple[QuizAttempt, ...]:
        """Return persisted quiz attempts from the existing attempt store."""
        return self.attempt_store.list_all()

    def generate_adaptive_quiz(
        self,
        query: str,
        *,
        question_count: int = DEFAULT_QUESTION_COUNT,
        available_topics: Sequence[str] = (),
        difficulty: Difficulty | None = None,
    ) -> Quiz:
        """Generate a quiz using stored attempts as adaptive history."""
        chunks = self._retrieve(query)
        return generate_adaptive_quiz(
            chunks,
            llm=self.rag.llm,
            attempts=self.attempt_store.list_all(),
            question_count=question_count,
            available_topics=available_topics,
            difficulty=difficulty,
            generator=self.quiz_generator,
        )

    def current_study_plan(
        self,
        *,
        available_topics: Sequence[str] = (),
        as_of: datetime | None = None,
    ) -> UnifiedStudyPlan:
        """Build the current study plan from stored attempts."""
        return self.planner.plan(
            attempts=self.attempt_store.list_all(),
            available_topics=available_topics,
            as_of=as_of,
        )

    def _retrieve(self, query: str) -> tuple[SimilarChunk, ...]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Question must not be empty")
        query_vector = self.rag.embeddings.embed(cleaned)
        return self.rag.store.search(query_vector, top_k=self.rag.top_k)


def create_study_assistant(settings: Settings | None = None) -> StudyAssistant:
    """Build a `StudyAssistant` from existing config-backed clients and stores."""
    return StudyAssistant(
        rag=create_rag_pipeline(settings),
        attempt_store=get_quiz_attempt_store(settings),
    )
