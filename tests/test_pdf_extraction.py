from pathlib import Path

import pytest

from study_assistant.pdf_extraction import (
    ExtractedDocument,
    InvalidPdfError,
    PdfNotFoundError,
    extract_pdf,
)
from tests.pdf_fixtures import minimal_pdf_bytes


def test_extract_pdf_returns_page_numbered_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(minimal_pdf_bytes(["Alpha page one", "Beta page two"]))

    document = extract_pdf(pdf_path)

    assert isinstance(document, ExtractedDocument)
    assert document.source_path == pdf_path.resolve()
    assert document.page_count == 2
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "Alpha page one" in document.pages[0].text
    assert "Beta page two" in document.pages[1].text
    assert "Beta page two" not in document.pages[0].text
    assert "Alpha page one" not in document.pages[1].text


def test_extract_pdf_keeps_blank_pages_in_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank-middle.pdf"
    pdf_path.write_bytes(minimal_pdf_bytes(["First", "", "Third"]))

    document = extract_pdf(pdf_path)

    assert document.page_count == 3
    assert document.pages[0].page_number == 1
    assert "First" in document.pages[0].text
    assert document.pages[1].page_number == 2
    assert document.pages[1].text.strip() == ""
    assert document.pages[2].page_number == 3
    assert "Third" in document.pages[2].text


def test_extract_pdf_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"

    with pytest.raises(PdfNotFoundError, match="not found"):
        extract_pdf(missing)


def test_extract_pdf_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(InvalidPdfError, match="Not a file"):
        extract_pdf(tmp_path)


def test_extract_pdf_garbage_file_is_invalid(tmp_path: Path) -> None:
    garbage = tmp_path / "not-a-pdf.pdf"
    garbage.write_bytes(b"this is not a pdf")

    with pytest.raises(InvalidPdfError, match="Invalid or unreadable PDF"):
        extract_pdf(garbage)
