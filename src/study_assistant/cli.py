"""Command-line entry point. Keep this thin; Q&A logic lives in pdf_qa."""

from __future__ import annotations

import logging
import sys

from study_assistant import __version__
from study_assistant.config import load_settings
from study_assistant.llm import complete
from study_assistant.llm.errors import LLMError
from study_assistant.pdf_extraction import PdfExtractionError
from study_assistant.pdf_qa import answer_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("study_assistant")

_HELP = """\
usage:
  python -m study_assistant
  python -m study_assistant PROMPT
  python -m study_assistant ask-pdf PDF_PATH QUESTION

commands:
  ask-pdf   Answer a question using a local PDF (full text in the prompt; not RAG)
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        return _show_status()
    if args[0] in {"-h", "--help", "help"}:
        print(_HELP)
        return 0
    if args[0] == "ask-pdf":
        return _run_ask_pdf(args[1:])
    return _run_prompt(args)


def _show_status() -> int:
    settings = load_settings()
    logger.info("Study Assistant v%s started", __version__)
    logger.info("env=%s provider=%s model=%s", settings.app_env, settings.llm_provider, settings.llm_model)
    logger.info("llm_base_url=%s", settings.llm_base_url)
    logger.info("data_dir=%s", settings.data_dir)
    return 0


def _run_ask_pdf(args: list[str]) -> int:
    if len(args) < 2:
        print("Error: ask-pdf requires a PDF path and a question.", file=sys.stderr)
        print(_HELP, file=sys.stderr)
        return 2

    pdf_path = args[0]
    question = " ".join(args[1:])
    try:
        answer = answer_from_pdf(pdf_path, question)
    except (PdfExtractionError, LLMError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Answer:")
    print(answer)
    return 0


def _run_prompt(args: list[str]) -> int:
    prompt = " ".join(args).strip()
    try:
        print(complete(prompt))
    except (LLMError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
