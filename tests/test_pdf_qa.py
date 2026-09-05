from pathlib import Path

import pytest

from study_assistant.llm.base import LLMClient
from study_assistant.pdf_extraction import ExtractedDocument, PageText
from study_assistant.pdf_qa import answer_from_pdf, build_pdf_qa_prompt


class FakeLLM(LLMClient):
    def __init__(self, response: str = "Answer from the document.") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _document(*page_texts: str, source: str = "notes.pdf") -> ExtractedDocument:
    pages = tuple(PageText(page_number=index, text=text) for index, text in enumerate(page_texts, start=1))
    return ExtractedDocument(source_path=Path(source), pages=pages)


def test_build_pdf_qa_prompt_includes_pages_and_grounding_rules() -> None:
    document = _document("Stacks use LIFO.", "Queues use FIFO.")

    prompt = build_pdf_qa_prompt(document, "What is a stack?")

    assert "ONLY the document text" in prompt
    assert "does not contain enough information" in prompt
    assert "--- Page 1 ---" in prompt
    assert "Stacks use LIFO." in prompt
    assert "--- Page 2 ---" in prompt
    assert "Queues use FIFO." in prompt
    assert "What is a stack?" in prompt
    assert "notes.pdf" in prompt


def test_build_pdf_qa_prompt_keeps_blank_page_numbers() -> None:
    document = _document("Intro", "", "Conclusion")

    prompt = build_pdf_qa_prompt(document, "Summarize the conclusion.")

    assert "--- Page 2 ---" in prompt
    assert "(no text extracted from this page)" in prompt
    assert "Conclusion" in prompt


def test_answer_from_pdf_uses_extractor_and_llm_client(tmp_path: Path) -> None:
    pdf_path = tmp_path / "lecture.pdf"
    extracted = _document("Photosynthesis converts light to chemical energy.", source=str(pdf_path))
    extract_calls: list[str | Path] = []

    def fake_extract(path: str | Path) -> ExtractedDocument:
        extract_calls.append(path)
        return extracted

    llm = FakeLLM(response="It converts light to chemical energy.")

    answer = answer_from_pdf(pdf_path, "What does photosynthesis do?", client=llm, extract=fake_extract)

    assert answer == "It converts light to chemical energy."
    assert extract_calls == [pdf_path]
    assert len(llm.prompts) == 1
    assert "--- Page 1 ---" in llm.prompts[0]
    assert "Photosynthesis converts light to chemical energy." in llm.prompts[0]
    assert "What does photosynthesis do?" in llm.prompts[0]
    assert "ONLY the document text" in llm.prompts[0]


def test_answer_from_pdf_rejects_empty_question() -> None:
    llm = FakeLLM()

    def fake_extract(path: str | Path) -> ExtractedDocument:
        raise AssertionError("extractor should not run for an empty question")

    with pytest.raises(ValueError, match="empty"):
        answer_from_pdf("notes.pdf", "   ", client=llm, extract=fake_extract)

    assert llm.prompts == []
