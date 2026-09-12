"""Command-line entry point. Keep this thin; logic lives in StudyAssistant."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from study_assistant import __version__
from study_assistant.config import load_settings
from study_assistant.llm import complete
from study_assistant.embeddings.errors import EmbeddingError
from study_assistant.llm.errors import LLMError
from study_assistant.pdf_extraction import PdfExtractionError
from study_assistant.pdf_qa import answer_from_pdf
from study_assistant.quiz import (
    DEFAULT_QUESTION_COUNT,
    DIFFICULTY_LEVELS,
    AnswerOption,
    QuestionSource,
    Quiz,
    QuizQuestion,
)
from study_assistant.quiz_attempts import QuizAttemptError
from study_assistant.quiz_evaluation import QuizEvaluationError, SubmittedAnswer
from study_assistant.service import StudyAssistant, create_study_assistant

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("study_assistant")

_HELP = """\
usage:
  python -m study_assistant
  python -m study_assistant PROMPT
  python -m study_assistant ask QUESTION
  python -m study_assistant quiz QUERY [--count N] [--difficulty easy|medium|hard]
  python -m study_assistant adaptive QUERY [--count N] [--topics t1,t2]
  python -m study_assistant plan
  python -m study_assistant evaluate QUIZ_JSON ANSWER [ANSWER ...]
  python -m study_assistant index PDF_PATH
  python -m study_assistant ask-pdf PDF_PATH QUESTION

commands:
  ask       Answer a question using RAG over the local vector store
  quiz      Generate a quiz from retrieved study material
  adaptive  Generate a quiz using stored attempt history
  plan      Show the current study plan
  evaluate  Score a saved quiz JSON file and record the attempt (- skips a question)
  index     Extract, embed, and store a local PDF (replaces prior chunks for that file)
  ask-pdf   Answer a question using a local PDF (full text in the prompt; not RAG)
"""

_SERVICE_COMMANDS = frozenset({"ask", "quiz", "adaptive", "plan", "evaluate", "index"})


def main(argv: list[str] | None = None, *, assistant: StudyAssistant | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        return _show_status()
    if args[0] in {"-h", "--help", "help"}:
        print(_HELP)
        return 0
    if args[0] == "ask-pdf":
        return _run_ask_pdf(args[1:])
    if args[0] in _SERVICE_COMMANDS:
        return _run_service_command(args[0], args[1:], assistant=assistant)
    return _run_prompt(args)


def _show_status() -> int:
    settings = load_settings()
    logger.info("Study Assistant v%s started", __version__)
    logger.info("env=%s provider=%s model=%s", settings.app_env, settings.llm_provider, settings.llm_model)
    logger.info("llm_base_url=%s", settings.llm_base_url)
    logger.info("data_dir=%s", settings.data_dir)
    return 0


def _run_service_command(
    command: str,
    args: list[str],
    *,
    assistant: StudyAssistant | None,
) -> int:
    try:
        service = assistant if assistant is not None else create_study_assistant()
        if command == "ask":
            return _cmd_ask(service, args)
        if command == "quiz":
            return _cmd_quiz(service, args)
        if command == "adaptive":
            return _cmd_adaptive(service, args)
        if command == "plan":
            return _cmd_plan(service, args)
        if command == "index":
            return _cmd_index(service, args)
        return _cmd_evaluate(service, args)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid quiz JSON: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (
        PdfExtractionError,
        EmbeddingError,
        LLMError,
        QuizEvaluationError,
        QuizAttemptError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_ask(assistant: StudyAssistant, args: list[str]) -> int:
    question = " ".join(args).strip()
    if not question:
        print("Error: ask requires a question.", file=sys.stderr)
        return 2
    result = assistant.ask(question)
    print("Answer:")
    print(result.answer)
    if result.sources:
        print()
        print("Sources:")
        for index, chunk in enumerate(result.sources, start=1):
            print(f"  [{index}] {chunk.source_path.as_posix()}, page {chunk.page_number}")
    return 0


def _cmd_quiz(assistant: StudyAssistant, args: list[str]) -> int:
    positional, options = _parse_options(args)
    query = " ".join(positional).strip()
    if not query:
        print("Error: quiz requires a query.", file=sys.stderr)
        return 2
    quiz = assistant.generate_quiz(
        query,
        question_count=options.get("count", DEFAULT_QUESTION_COUNT),
        difficulty=options.get("difficulty", "medium"),
    )
    _print_quiz(quiz)
    return 0


def _cmd_adaptive(assistant: StudyAssistant, args: list[str]) -> int:
    positional, options = _parse_options(args)
    query = " ".join(positional).strip()
    if not query:
        print("Error: adaptive requires a query.", file=sys.stderr)
        return 2
    quiz = assistant.generate_adaptive_quiz(
        query,
        question_count=options.get("count", DEFAULT_QUESTION_COUNT),
        available_topics=options.get("topics", ()),
        difficulty=options.get("difficulty"),
    )
    _print_quiz(quiz)
    return 0


def _cmd_plan(assistant: StudyAssistant, args: list[str]) -> int:
    positional, options = _parse_options(args)
    if positional:
        print("Error: plan does not take a query.", file=sys.stderr)
        return 2
    plan = assistant.current_study_plan(available_topics=options.get("topics", ()))
    if not plan.items:
        print("No study plan items.")
        return 0
    print("Study plan:")
    for item in plan.items:
        print(
            f"{item.priority}. {item.topic}  "
            f"difficulty={item.recommended_difficulty}  "
            f"review={item.next_review.isoformat()}"
        )
        print(f"   {item.reason}")
    return 0


def _cmd_index(assistant: StudyAssistant, args: list[str]) -> int:
    pdf_path = " ".join(args).strip()
    if not pdf_path:
        print("Error: index requires a PDF path.", file=sys.stderr)
        return 2
    result = assistant.index_pdf(pdf_path)
    print(f"Indexed {result.source_path.as_posix()}")
    print(f"Stored chunks: {result.chunk_count}")
    return 0


def _cmd_evaluate(assistant: StudyAssistant, args: list[str]) -> int:
    if not args:
        print("Error: evaluate requires a quiz JSON file and answers.", file=sys.stderr)
        return 2
    quiz_path = Path(args[0])
    if not quiz_path.is_file():
        print(f"Error: quiz file not found: {quiz_path}", file=sys.stderr)
        return 2
    raw_answers = args[1:]
    if not raw_answers:
        print("Error: evaluate requires at least one answer (use - to skip).", file=sys.stderr)
        return 2
    quiz = _load_quiz(quiz_path)
    answers = tuple(
        SubmittedAnswer(index, None if token == "-" else token)
        for index, token in enumerate(raw_answers)
    )
    result, _attempt = assistant.evaluate_and_record(quiz, answers)
    print(
        f"Score: {result.correct_count}/{result.total_questions} "
        f"({result.score_percent:.1f}%)"
    )
    print(
        f"correct={result.correct_count}  "
        f"incorrect={result.incorrect_count}  "
        f"unanswered={result.unanswered_count}"
    )
    for item in result.question_results:
        status = "correct" if item.is_correct else ("unanswered" if item.is_unanswered else "incorrect")
        selected = item.selected_label if item.selected_label is not None else "-"
        print(f"  Q{item.question_index + 1}: {status} (selected {selected}, answer {item.correct_label})")
    return 0


def _print_quiz(quiz: Quiz) -> None:
    print(f"Quiz {quiz.quiz_id}  difficulty={quiz.difficulty}")
    for index, question in enumerate(quiz.questions, start=1):
        topic = f"  [{question.topic}]" if question.topic else ""
        print(f"\n{index}.{topic} {question.question}")
        for option in question.options:
            print(f"  {option.label}. {option.text}")
    print()
    print("Quiz JSON:")
    print(json.dumps(_quiz_to_dict(quiz), indent=2))


def _quiz_to_dict(quiz: Quiz) -> dict[str, Any]:
    return {
        "quiz_id": quiz.quiz_id,
        "difficulty": quiz.difficulty,
        "questions": [
            {
                "question": question.question,
                "options": [{"label": option.label, "text": option.text} for option in question.options],
                "correct_label": question.correct_label,
                "explanation": question.explanation,
                "topic": question.topic,
                "sources": [
                    {
                        "citation_index": source.citation_index,
                        "chunk_id": source.chunk_id,
                        "source_path": source.source_path.as_posix(),
                        "page_number": source.page_number,
                    }
                    for source in question.sources
                ],
            }
            for question in quiz.questions
        ],
    }


def _load_quiz(path: Path) -> Quiz:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Quiz JSON must be an object")
    questions_raw = payload.get("questions")
    if not isinstance(questions_raw, list):
        raise ValueError("Quiz JSON must contain a questions list")
    questions = tuple(_question_from_dict(item, index=index) for index, item in enumerate(questions_raw, start=1))
    difficulty = str(payload.get("difficulty", "medium"))
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError("difficulty must be one of: easy, medium, hard")
    quiz_id = str(payload.get("quiz_id", "")).strip() or "cli-quiz"
    return Quiz(questions=questions, difficulty=difficulty, quiz_id=quiz_id)  # type: ignore[arg-type]


def _question_from_dict(item: object, *, index: int) -> QuizQuestion:
    if not isinstance(item, dict):
        raise ValueError(f"Question {index} must be an object")
    options_raw = item.get("options")
    if not isinstance(options_raw, list):
        raise ValueError(f"Question {index} must have options")
    sources_raw = item.get("sources")
    if not isinstance(sources_raw, list):
        sources_raw = []
    return QuizQuestion(
        question=str(item.get("question", "")).strip(),
        options=tuple(
            AnswerOption(label=str(option.get("label", "")), text=str(option.get("text", "")))
            for option in options_raw
            if isinstance(option, dict)
        ),
        correct_label=str(item.get("correct_label", "")).strip(),
        explanation=str(item.get("explanation", "")).strip(),
        sources=tuple(_source_from_dict(source) for source in sources_raw if isinstance(source, dict)),
        topic=str(item.get("topic", "")).strip(),
    )


def _source_from_dict(item: dict[str, Any]) -> QuestionSource:
    return QuestionSource(
        citation_index=int(item.get("citation_index", 1)),
        chunk_id=str(item.get("chunk_id", "")),
        source_path=Path(str(item.get("source_path", "unknown"))),
        page_number=int(item.get("page_number", 1)),
    )


def _parse_options(args: list[str]) -> tuple[list[str], dict[str, Any]]:
    positional: list[str] = []
    options: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--count", "--difficulty", "--topics"}:
            if index + 1 >= len(args):
                raise ValueError(f"{token} requires a value")
            value = args[index + 1]
            if token == "--count":
                options["count"] = int(value)
            elif token == "--difficulty":
                if value not in DIFFICULTY_LEVELS:
                    raise ValueError("difficulty must be one of: easy, medium, hard")
                options["difficulty"] = value
            else:
                options["topics"] = tuple(part.strip() for part in value.split(",") if part.strip())
            index += 2
            continue
        if token.startswith("-"):
            raise ValueError(f"Unknown option: {token}")
        positional.append(token)
        index += 1
    return positional, options


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
