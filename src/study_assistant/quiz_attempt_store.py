"""Persistent quiz attempt history backed by SQLite.

This module does not generate quizzes, score answers, embed text, or call an LLM.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from study_assistant.config import Settings, load_settings
from study_assistant.quiz import DIFFICULTY_LEVELS, Difficulty, QuestionSource
from study_assistant.quiz_attempts import QuizAttempt
from study_assistant.quiz_evaluation import QuestionResult, QuizEvaluation

_CREATE_ATTEMPTS_TABLE = """
CREATE TABLE IF NOT EXISTS quiz_attempts (
    attempt_id TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    correct_count INTEGER NOT NULL,
    score_percent REAL NOT NULL,
    question_results TEXT NOT NULL
)
"""

_CREATE_QUIZ_INDEX = """
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON quiz_attempts (quiz_id)
"""


class QuizAttemptStoreError(Exception):
    """Raised when stored attempt data cannot be read, written, or decoded."""


class SqliteQuizAttemptStore:
    """SQLite-backed attempt history. Creates the schema on first use."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def add(self, attempt: QuizAttempt) -> str:
        """Insert one attempt. Raises if `attempt_id` is already stored."""
        attempt_id = attempt.attempt_id.strip()
        if not attempt_id:
            raise QuizAttemptStoreError("attempt_id must be a non-empty string")
        quiz_id = attempt.quiz_id.strip()
        if not quiz_id:
            raise QuizAttemptStoreError("quiz_id must be a non-empty string")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO quiz_attempts (
                        attempt_id, quiz_id, timestamp, difficulty,
                        total_questions, correct_count, score_percent, question_results
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        quiz_id,
                        _timestamp_to_iso(attempt.timestamp),
                        attempt.difficulty,
                        attempt.evaluation.total_questions,
                        attempt.evaluation.correct_count,
                        attempt.evaluation.score_percent,
                        serialize_question_results(attempt.question_results),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise QuizAttemptStoreError(f"Attempt {attempt_id!r} already exists") from exc
        return attempt_id

    def get(self, attempt_id: str) -> QuizAttempt | None:
        """Return one attempt by id, or None if it is not stored."""
        row = self._connection.execute(
            """
            SELECT attempt_id, quiz_id, timestamp, difficulty,
                   total_questions, correct_count, score_percent, question_results
            FROM quiz_attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_attempt(row)

    def list_for_quiz(self, quiz_id: str) -> tuple[QuizAttempt, ...]:
        """Return attempts for a quiz, oldest first."""
        rows = self._connection.execute(
            """
            SELECT attempt_id, quiz_id, timestamp, difficulty,
                   total_questions, correct_count, score_percent, question_results
            FROM quiz_attempts
            WHERE quiz_id = ?
            ORDER BY timestamp ASC, attempt_id ASC
            """,
            (quiz_id,),
        ).fetchall()
        return tuple(_row_to_attempt(row) for row in rows)

    def list_all(self) -> tuple[QuizAttempt, ...]:
        """Return every stored attempt, oldest first."""
        rows = self._connection.execute(
            """
            SELECT attempt_id, quiz_id, timestamp, difficulty,
                   total_questions, correct_count, score_percent, question_results
            FROM quiz_attempts
            ORDER BY timestamp ASC, attempt_id ASC
            """
        ).fetchall()
        return tuple(_row_to_attempt(row) for row in rows)

    def count(self) -> int:
        """Return the number of stored attempts."""
        row = self._connection.execute("SELECT COUNT(*) AS n FROM quiz_attempts").fetchone()
        return int(row["n"]) if row is not None else 0

    def delete(self, attempt_id: str) -> bool:
        """Delete one attempt. Returns True if a row was removed."""
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM quiz_attempts WHERE attempt_id = ?",
                (attempt_id,),
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteQuizAttemptStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self._connection:
            self._connection.execute(_CREATE_ATTEMPTS_TABLE)
            self._connection.execute(_CREATE_QUIZ_INDEX)


def get_quiz_attempt_store(settings: Settings | None = None) -> SqliteQuizAttemptStore:
    """Open the SQLite attempt store at the configured path, creating it if needed."""
    resolved = settings if settings is not None else load_settings()
    return SqliteQuizAttemptStore(resolved.quiz_attempt_store_path)


def serialize_question_results(results: Sequence[QuestionResult]) -> str:
    """Encode per-question outcomes as canonical JSON."""
    payload = [_question_result_to_dict(item) for item in results]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deserialize_question_results(raw: str) -> tuple[QuestionResult, ...]:
    """Decode canonical JSON into per-question outcomes."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QuizAttemptStoreError("Stored question results are not valid JSON") from exc
    if not isinstance(payload, list):
        raise QuizAttemptStoreError("Stored question results must be a JSON list")
    return tuple(_dict_to_question_result(item, index=index) for index, item in enumerate(payload))


def _question_result_to_dict(item: QuestionResult) -> dict[str, object]:
    return {
        "question_index": item.question_index,
        "selected_label": item.selected_label,
        "correct_label": item.correct_label,
        "is_correct": item.is_correct,
        "is_unanswered": item.is_unanswered,
        "explanation": item.explanation,
        "topic": item.topic,
        "sources": [
            {
                "citation_index": source.citation_index,
                "chunk_id": source.chunk_id,
                "source_path": source.source_path.as_posix(),
                "page_number": source.page_number,
            }
            for source in item.sources
        ],
    }


def _dict_to_question_result(item: object, *, index: int) -> QuestionResult:
    if not isinstance(item, dict):
        raise QuizAttemptStoreError(f"Question result {index} must be an object")
    sources_raw = item.get("sources")
    if not isinstance(sources_raw, list):
        raise QuizAttemptStoreError(f"Question result {index} sources must be a list")
    return QuestionResult(
        question_index=_require_int(item.get("question_index"), field=f"Question result {index} question_index"),
        selected_label=_optional_label(item.get("selected_label"), field=f"Question result {index} selected_label"),
        correct_label=_require_text(item.get("correct_label"), field=f"Question result {index} correct_label"),
        is_correct=_require_bool(item.get("is_correct"), field=f"Question result {index} is_correct"),
        is_unanswered=_require_bool(item.get("is_unanswered"), field=f"Question result {index} is_unanswered"),
        explanation=_require_text(item.get("explanation"), field=f"Question result {index} explanation"),
        sources=tuple(_dict_to_source(source, question_index=index) for source in sources_raw),
        topic=_optional_topic(item.get("topic"), field=f"Question result {index} topic"),
    )


def _dict_to_source(item: object, *, question_index: int) -> QuestionSource:
    if not isinstance(item, dict):
        raise QuizAttemptStoreError(f"Question result {question_index} source must be an object")
    return QuestionSource(
        citation_index=_require_int(item.get("citation_index"), field=f"Question result {question_index} citation_index"),
        chunk_id=_require_text(item.get("chunk_id"), field=f"Question result {question_index} chunk_id"),
        source_path=Path(_require_text(item.get("source_path"), field=f"Question result {question_index} source_path")),
        page_number=_require_int(item.get("page_number"), field=f"Question result {question_index} page_number"),
    )


def _row_to_attempt(row: sqlite3.Row) -> QuizAttempt:
    results = deserialize_question_results(row["question_results"])
    difficulty = _parse_difficulty(row["difficulty"])
    unanswered_count = sum(1 for item in results if item.is_unanswered)
    incorrect_count = sum(1 for item in results if not item.is_unanswered and not item.is_correct)
    evaluation = QuizEvaluation(
        total_questions=int(row["total_questions"]),
        correct_count=int(row["correct_count"]),
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        score_percent=float(row["score_percent"]),
        question_results=results,
    )
    return QuizAttempt(
        quiz_id=row["quiz_id"],
        timestamp=_timestamp_from_iso(row["timestamp"]),
        evaluation=evaluation,
        difficulty=difficulty,
        question_results=results,
        attempt_id=row["attempt_id"],
    )


def _parse_difficulty(value: object) -> Difficulty:
    if isinstance(value, str):
        for level in DIFFICULTY_LEVELS:
            if value == level:
                return level
    raise QuizAttemptStoreError(f"Stored difficulty {value!r} is invalid")


def _timestamp_to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _timestamp_from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuizAttemptStoreError(f"{field} must be an integer")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise QuizAttemptStoreError(f"{field} must be a boolean")
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuizAttemptStoreError(f"{field} must be a non-empty string")
    return value


def _optional_label(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise QuizAttemptStoreError(f"{field} must be a string or null")
    return value


def _optional_topic(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise QuizAttemptStoreError(f"{field} must be a string")
    return value.strip()
