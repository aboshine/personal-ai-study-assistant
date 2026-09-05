"""Entry point: `python -m study_assistant`."""

from __future__ import annotations

import logging
import sys

from study_assistant import __version__
from study_assistant.config import load_settings
from study_assistant.llm import complete

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("study_assistant")


def main() -> int:
    settings = load_settings()
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        print(complete(prompt, settings=settings))
        return 0

    logger.info("Study Assistant v%s started", __version__)
    logger.info("env=%s provider=%s model=%s", settings.app_env, settings.llm_provider, settings.llm_model)
    logger.info("llm_base_url=%s", settings.llm_base_url)
    logger.info("data_dir=%s", settings.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
