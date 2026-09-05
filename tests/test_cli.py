from study_assistant.cli import main
from study_assistant.llm.errors import LLMConnectionError
from study_assistant.pdf_extraction import PdfNotFoundError


def test_ask_pdf_prints_answer(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "study_assistant.cli.answer_from_pdf",
        lambda pdf_path, question: f"{pdf_path}:{question}:Stacks use LIFO.",
    )

    code = main(["ask-pdf", "notes.pdf", "What", "is", "a", "stack?"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "Answer:\nnotes.pdf:What is a stack?:Stacks use LIFO.\n"
    assert captured.err == ""


def test_ask_pdf_requires_path_and_question(capsys) -> None:
    code = main(["ask-pdf", "notes.pdf"])

    captured = capsys.readouterr()
    assert code == 2
    assert "PDF path and a question" in captured.err
    assert captured.out == ""


def test_ask_pdf_prints_service_error_without_traceback(capsys, monkeypatch) -> None:
    def fake_answer(pdf_path: str, question: str) -> str:
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")

    monkeypatch.setattr("study_assistant.cli.answer_from_pdf", fake_answer)

    code = main(["ask-pdf", "missing.pdf", "What is a stack?"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Error: PDF not found: missing.pdf" in captured.err
    assert "Traceback" not in captured.err


def test_ask_pdf_prints_llm_error_without_traceback(capsys, monkeypatch) -> None:
    def fake_answer(pdf_path: str, question: str) -> str:
        raise LLMConnectionError("Cannot connect to Ollama at http://127.0.0.1:11434. Is Ollama running?")

    monkeypatch.setattr("study_assistant.cli.answer_from_pdf", fake_answer)

    code = main(["ask-pdf", "notes.pdf", "What is a stack?"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Error: Cannot connect to Ollama" in captured.err
    assert "Traceback" not in captured.err


def test_help_prints_ask_pdf_usage(capsys) -> None:
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "ask-pdf PDF_PATH QUESTION" in captured.out
