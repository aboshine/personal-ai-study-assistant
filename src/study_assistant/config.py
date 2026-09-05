"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets stay in the environment, not in code."""

    app_env: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    embedding_provider: str
    embedding_model: str
    data_dir: Path


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """Load settings from a `.env` file (if present) and the process environment."""
    env_file = dotenv_path if dotenv_path is not None else _PROJECT_ROOT / ".env"
    load_dotenv(env_file)

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = (_PROJECT_ROOT / data_dir).resolve()

    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "llama3.2"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        data_dir=data_dir,
    )
