"""Extract text from local PDF files, keeping page boundaries.

This module is a building block for later RAG. It only reads a file and
returns page-numbered text; it does not chunk, embed, or store anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError, PdfStreamError


class PdfExtractionError(Exception):
    """Base error for PDF extraction failures."""


class PdfNotFoundError(PdfExtractionError):
    """Raised when the given path does not exist."""


class InvalidPdfError(PdfExtractionError):
    """Raised when the path is not a readable, unencrypted PDF."""


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page. `page_number` is 1-based."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Full extraction result with one entry per page, including blank pages."""

    source_path: Path
    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract text from each page of a local PDF file.

    Raises:
        PdfNotFoundError: the path does not exist.
        InvalidPdfError: the path is not a file, is encrypted, or is not a valid PDF.
    """
    pdf_path = Path(path)
    reader = _open_pdf(pdf_path)

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        pages.append(PageText(page_number=index, text=raw if raw else ""))

    return ExtractedDocument(source_path=pdf_path.resolve(), pages=tuple(pages))


def _open_pdf(pdf_path: Path) -> PdfReader:
    if not pdf_path.exists():
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise InvalidPdfError(f"Not a file: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, PdfStreamError, OSError) as exc:
        raise InvalidPdfError(f"Invalid or unreadable PDF: {pdf_path}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise InvalidPdfError(f"Encrypted PDF cannot be opened: {pdf_path}") from exc
        if not unlocked:
            raise InvalidPdfError(f"Encrypted PDF cannot be opened: {pdf_path}")

    try:
        _ = len(reader.pages)
    except (FileNotDecryptedError, PdfReadError, PdfStreamError) as exc:
        raise InvalidPdfError(f"Invalid or unreadable PDF: {pdf_path}") from exc

    return reader
