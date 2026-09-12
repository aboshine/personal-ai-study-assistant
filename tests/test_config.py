from pathlib import Path

from study_assistant.config import load_settings


def test_load_settings_uses_env_file(tmp_path: Path, monkeypatch: object) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=test",
                "LLM_PROVIDER=ollama",
                "LLM_MODEL=test-model",
                "LLM_BASE_URL=http://127.0.0.1:11434",
                "EMBEDDING_PROVIDER=ollama",
                "EMBEDDING_MODEL=nomic-embed-text",
                "DATA_DIR=./course-data",
                "VECTOR_STORE_PATH=./course-data/vector_store.sqlite",
                "QUIZ_ATTEMPT_STORE_PATH=./course-data/quiz_attempts.sqlite",
                "STUDY_PLAN_STORE_PATH=./course-data/study_plans.sqlite",
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "APP_ENV",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "DATA_DIR",
        "VECTOR_STORE_PATH",
        "QUIZ_ATTEMPT_STORE_PATH",
        "STUDY_PLAN_STORE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(dotenv_path=env_file)

    assert settings.app_env == "test"
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "test-model"
    assert settings.llm_base_url == "http://127.0.0.1:11434"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.data_dir.is_absolute()
    assert settings.data_dir.name == "course-data"
    assert settings.vector_store_path.is_absolute()
    assert settings.vector_store_path.name == "vector_store.sqlite"
    assert settings.quiz_attempt_store_path.is_absolute()
    assert settings.quiz_attempt_store_path.name == "quiz_attempts.sqlite"
    assert settings.study_plan_store_path.is_absolute()
    assert settings.study_plan_store_path.name == "study_plans.sqlite"


def test_load_settings_prefers_process_environment(tmp_path: Path, monkeypatch: object) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-process")

    settings = load_settings(dotenv_path=env_file)

    assert settings.llm_model == "from-process"
