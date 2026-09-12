from pathlib import Path

import pytest

from study_assistant.config import Settings
from study_assistant.study_plan import StudyPlan, generate_study_plan
from study_assistant.study_plan_store import (
    SqliteStudyPlanStore,
    StudyPlanStoreError,
    get_study_plan_store,
)
from study_assistant.topic_tracking import TopicPerformance


def _perf(topic: str, accuracy: float) -> TopicPerformance:
    answered = 4 if accuracy else 0
    correct = round(accuracy / 100.0 * answered) if answered else 0
    return TopicPerformance(
        topic=topic,
        attempt_count=1,
        questions_answered=answered,
        correct_count=correct,
        accuracy=accuracy,
    )


def _plan(plan_id: str, *topics_and_accuracy: tuple[str, float]) -> StudyPlan:
    items = generate_study_plan(tuple(_perf(topic, accuracy) for topic, accuracy in topics_and_accuracy))
    return StudyPlan(plan_id=plan_id, items=items)


def _settings(db_path: Path) -> Settings:
    return Settings(
        app_env="test",
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_base_url="http://127.0.0.1:11434",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        data_dir=db_path.parent,
        vector_store_path=db_path.parent / "vector_store.sqlite",
        quiz_attempt_store_path=db_path.parent / "quiz_attempts.sqlite",
        study_plan_store_path=db_path,
    )


def test_empty_store(tmp_path: Path) -> None:
    with SqliteStudyPlanStore(tmp_path / "plans.sqlite") as store:
        assert store.count() == 0
        assert store.list_plans() == ()
        assert store.get("missing") is None
        assert store.delete("missing") is False


def test_save_get_round_trip_preserves_plan_data(tmp_path: Path) -> None:
    plan = _plan("plan-a", ("Stacks", 90.0), ("Queues", 25.0))
    with SqliteStudyPlanStore(tmp_path / "plans.sqlite") as store:
        assert store.save(plan) == "plan-a"
        loaded = store.get("plan-a")
    assert loaded == plan
    assert loaded is not None
    assert [item.topic for item in loaded.items] == ["Queues", "Stacks"]
    assert loaded.items[0].priority == 1
    assert loaded.items[0].recommended_difficulty == "easy"
    assert "Low accuracy" in loaded.items[0].reason
    assert loaded.items[1].recommended_difficulty == "hard"


def test_persists_across_separate_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "plans.sqlite"
    plan = _plan("persist-1", ("Heaps", 60.0))
    writer = SqliteStudyPlanStore(db_path)
    writer.save(plan)
    writer.close()

    reader = SqliteStudyPlanStore(db_path)
    assert reader.get("persist-1") == plan
    assert reader.count() == 1
    reader.close()


def test_multiple_plans_and_listing(tmp_path: Path) -> None:
    first = _plan("b-plan", ("Queues", 10.0))
    second = _plan("a-plan", ("Stacks", 80.0))
    empty = StudyPlan(plan_id="empty", items=())
    with SqliteStudyPlanStore(tmp_path / "plans.sqlite") as store:
        store.save(first)
        store.save(second)
        store.save(empty)
        listed = store.list_plans()
        assert [plan.plan_id for plan in listed] == ["a-plan", "b-plan", "empty"]
        assert listed[0] == second
        assert listed[2].items == ()
        assert store.count() == 3


def test_delete_plan(tmp_path: Path) -> None:
    keep = _plan("keep", ("Stacks", 70.0))
    drop = _plan("drop", ("Queues", 20.0))
    with SqliteStudyPlanStore(tmp_path / "plans.sqlite") as store:
        store.save(keep)
        store.save(drop)
        assert store.delete("drop") is True
        assert store.delete("drop") is False
        assert store.get("drop") is None
        assert store.get("keep") == keep
        assert store.count() == 1


def test_duplicate_plan_id_is_rejected(tmp_path: Path) -> None:
    first = _plan("dup", ("Queues", 25.0))
    second = _plan("dup", ("Stacks", 90.0))
    with SqliteStudyPlanStore(tmp_path / "plans.sqlite") as store:
        store.save(first)
        with pytest.raises(StudyPlanStoreError, match="already exists"):
            store.save(second)
        assert store.count() == 1
        assert store.get("dup") == first


def test_get_study_plan_store_uses_settings_path(tmp_path: Path) -> None:
    db_path = tmp_path / "from-settings.sqlite"
    plan = _plan("via-settings", ("Graphs", 0.0))
    store = get_study_plan_store(_settings(db_path))
    store.save(plan)
    store.close()
    reopened = SqliteStudyPlanStore(db_path)
    assert reopened.get("via-settings") == plan
    reopened.close()
