"""Split extracted PDF pages into overlapping text chunks.

Chunks stay on a single page so later retrieval can keep page metadata.
This module does not embed, store, or retrieve anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from study_assistant.pdf_extraction import ExtractedDocument

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class DocumentChunk:
    """One retrieval-sized piece of a document page."""

    chunk_id: str
    text: str
    source_path: Path
    page_number: int


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[DocumentChunk, ...]:
    """Split each page of `document` into deterministic character windows.

    Pages are chunked independently. `chunk_size` and `overlap` are measured
    in characters. `overlap` must be smaller than `chunk_size`.
    """
    _validate_chunk_params(chunk_size, overlap)

    chunks: list[DocumentChunk] = []
    for page in document.pages:
        page_chunks = _chunk_page_text(
            text=page.text,
            source_path=document.source_path,
            page_number=page.page_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(page_chunks)
    return tuple(chunks)


def make_chunk_id(source_path: Path, page_number: int, chunk_index: int) -> str:
    """Return a stable id for a given source file, page, and on-page index."""
    return f"{source_path.name}:p{page_number}:c{chunk_index}"


def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap must be at least 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _chunk_page_text(
    *,
    text: str,
    source_path: Path,
    page_number: int,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    if not text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 1
    text_length = len(text)

    while start < text_length:
        piece = text[start : start + chunk_size]
        if piece.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source_path, page_number, chunk_index),
                    text=piece,
                    source_path=source_path,
                    page_number=page_number,
                )
            )
            chunk_index += 1
        if start + chunk_size >= text_length:
            break
        start += step

    return chunks
