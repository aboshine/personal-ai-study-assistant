"""Answer a question using extracted PDF text and an LLMClient.

This is not RAG: the full extracted document is placed in the prompt.
Citations, embeddings, and retrieval are intentionally omitted.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from study_assistant.llm import LLMClient, get_llm_client
from study_assistant.pdf_extraction import ExtractedDocument, extract_pdf

ExtractFn = Callable[[str | Path], ExtractedDocument]


def answer_from_pdf(
    pdf_path: str | Path,
    question: str,
    *,
    client: LLMClient | None = None,
    extract: ExtractFn = extract_pdf,
) -> str:
    """Extract a local PDF and ask the LLM to answer from that text only."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question must not be empty")

    document = extract(pdf_path)
    prompt = build_pdf_qa_prompt(document, cleaned_question)
    llm = client if client is not None else get_llm_client()
    return llm.complete(prompt)


def build_pdf_qa_prompt(document: ExtractedDocument, question: str) -> str:
    """Build a page-numbered document prompt for later citation support."""
    pages_block = "\n\n".join(_format_page(page.page_number, page.text) for page in document.pages)
    if not pages_block:
        pages_block = "(This document has no pages.)"

    return (
        "You are a study assistant.\n"
        "Answer the user's question using ONLY the document text below.\n"
        "If the document does not contain enough information to answer, say so clearly.\n"
        "Do not use outside knowledge.\n"
        "Page numbers are included with each page so answers can cite pages later; "
        "do not invent page numbers.\n"
        "\n"
        f"Document: {document.source_path.name}\n"
        "\n"
        f"{pages_block}\n"
        "\n"
        "Question:\n"
        f"{question.strip()}\n"
    )


def _format_page(page_number: int, text: str) -> str:
    body = text if text.strip() else "(no text extracted from this page)"
    return f"--- Page {page_number} ---\n{body}"
