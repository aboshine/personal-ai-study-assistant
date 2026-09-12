"""Persistent study plans backed by SQLite.

This module does not generate plans, score quizzes, or call an LLM.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from study_assistant.config import Settings, load_settings
from study_assistant.quiz import DIFFICULTY_LEVELS, Difficulty
from study_assistant.study_plan import StudyPlan, StudyPlanItem

_CREATE_PLANS_TABLE = """
CREATE TABLE IF NOT EXISTS study_plans (
    plan_id TEXT PRIMARY KEY
)
"""

_CREATE_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS study_plan_items (
    plan_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    priority INTEGER NOT NULL,
    recommended_difficulty TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (plan_id, topic),
    FOREIGN KEY (plan_id) REFERENCES study_plans(plan_id) ON DELETE CASCADE
)
"""


class StudyPlanStoreError(Exception):
    """Raised when stored study-plan data cannot be read or written."""


class SqliteStudyPlanStore:
    """SQLite-backed study-plan store. Creates the schema on first use."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def save(self, plan: StudyPlan) -> str:
        """Insert one plan. Raises if `plan_id` is already stored."""
        plan_id = plan.plan_id.strip()
        if not plan_id:
            raise StudyPlanStoreError("plan_id must be a non-empty string")
        rows = [_item_row(plan_id, item) for item in plan.items]
        try:
            with self._connection:
                self._connection.execute("INSERT INTO study_plans (plan_id) VALUES (?)", (plan_id,))
                self._connection.executemany(
                    """
                    INSERT INTO study_plan_items (
                        plan_id, topic, priority, recommended_difficulty, reason
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.IntegrityError as exc:
            raise StudyPlanStoreError(f"Plan {plan_id!r} already exists") from exc
        return plan_id

    def get(self, plan_id: str) -> StudyPlan | None:
        """Return one plan by id, or None if it is not stored."""
        header = self._connection.execute(
            "SELECT plan_id FROM study_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if header is None:
            return None
        return StudyPlan(plan_id=header["plan_id"], items=_load_items(self._connection, plan_id))

    def list_plans(self) -> tuple[StudyPlan, ...]:
        """Return every stored plan, ordered by plan_id."""
        rows = self._connection.execute(
            "SELECT plan_id FROM study_plans ORDER BY plan_id ASC"
        ).fetchall()
        return tuple(
            StudyPlan(plan_id=row["plan_id"], items=_load_items(self._connection, row["plan_id"]))
            for row in rows
        )

    def count(self) -> int:
        """Return the number of stored plans."""
        row = self._connection.execute("SELECT COUNT(*) AS n FROM study_plans").fetchone()
        return int(row["n"]) if row is not None else 0

    def delete(self, plan_id: str) -> bool:
        """Delete one plan and its items. Returns True if a plan was removed."""
        with self._connection:
            cursor = self._connection.execute("DELETE FROM study_plans WHERE plan_id = ?", (plan_id,))
        return cursor.rowcount > 0

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteStudyPlanStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self._connection:
            self._connection.execute(_CREATE_PLANS_TABLE)
            self._connection.execute(_CREATE_ITEMS_TABLE)


def get_study_plan_store(settings: Settings | None = None) -> SqliteStudyPlanStore:
    """Open the SQLite study-plan store at the configured path, creating it if needed."""
    resolved = settings if settings is not None else load_settings()
    return SqliteStudyPlanStore(resolved.study_plan_store_path)


def _item_row(plan_id: str, item: StudyPlanItem) -> tuple[str, str, int, str, str]:
    topic = item.topic.strip()
    if not topic:
        raise StudyPlanStoreError("topic must be a non-empty string")
    if not isinstance(item.priority, int) or isinstance(item.priority, bool) or item.priority < 1:
        raise StudyPlanStoreError("priority must be an integer >= 1")
    difficulty = _parse_difficulty(item.recommended_difficulty)
    reason = item.reason.strip()
    if not reason:
        raise StudyPlanStoreError("reason must be a non-empty string")
    return (plan_id, topic, item.priority, difficulty, reason)


def _load_items(connection: sqlite3.Connection, plan_id: str) -> tuple[StudyPlanItem, ...]:
    rows = connection.execute(
        """
        SELECT topic, priority, recommended_difficulty, reason
        FROM study_plan_items
        WHERE plan_id = ?
        ORDER BY priority ASC, topic ASC
        """,
        (plan_id,),
    ).fetchall()
    return tuple(_row_to_item(row) for row in rows)


def _row_to_item(row: sqlite3.Row) -> StudyPlanItem:
    return StudyPlanItem(
        topic=row["topic"],
        priority=int(row["priority"]),
        recommended_difficulty=_parse_difficulty(row["recommended_difficulty"]),
        reason=row["reason"],
    )


def _parse_difficulty(value: object) -> Difficulty:
    if isinstance(value, str):
        for level in DIFFICULTY_LEVELS:
            if value == level:
                return level
    raise StudyPlanStoreError(f"Stored difficulty {value!r} is invalid")
