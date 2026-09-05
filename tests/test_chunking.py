from pathlib import Path

import pytest

from study_assistant.chunking import chunk_document, make_chunk_id
from study_assistant.pdf_extraction import ExtractedDocument, PageText


def _document(*page_texts: str, source: str = "notes.pdf") -> ExtractedDocument:
    pages = tuple(PageText(page_number=index, text=text) for index, text in enumerate(page_texts, start=1))
    return ExtractedDocument(source_path=Path(source), pages=pages)


def test_short_page_becomes_one_chunk() -> None:
    document = _document("Stacks use LIFO.")

    chunks = chunk_document(document, chunk_size=50, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Stacks use LIFO."
    assert chunks[0].page_number == 1
    assert chunks[0].source_path == Path("notes.pdf")
    assert chunks[0].chunk_id == "notes.pdf:p1:c1"


def test_empty_pages_produce_no_chunks() -> None:
    document = _document("Intro", "   ", "Conclusion")

    chunks = chunk_document(document, chunk_size=50, overlap=0)

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert chunks[0].text == "Intro"
    assert chunks[1].text == "Conclusion"


def test_long_page_splits_into_fixed_windows() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    document = _document(text)

    chunks = chunk_document(document, chunk_size=10, overlap=0)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 1]
    assert [chunk.chunk_id for chunk in chunks] == [
        "notes.pdf:p1:c1",
        "notes.pdf:p1:c2",
        "notes.pdf:p1:c3",
    ]


def test_overlap_repeats_characters_from_previous_chunk() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    document = _document(text)

    chunks = chunk_document(document, chunk_size=10, overlap=3)

    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text == "hijklmnopq"
    assert chunks[2].text == "opqrstuvwx"
    assert chunks[3].text == "vwxyz"
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]


def test_page_boundaries_are_not_merged() -> None:
    document = _document("AAAABBBBCCCC", "XXXXYYYYZZZZ")

    chunks = chunk_document(document, chunk_size=4, overlap=0)

    page_one = [chunk for chunk in chunks if chunk.page_number == 1]
    page_two = [chunk for chunk in chunks if chunk.page_number == 2]
    assert [chunk.text for chunk in page_one] == ["AAAA", "BBBB", "CCCC"]
    assert [chunk.text for chunk in page_two] == ["XXXX", "YYYY", "ZZZZ"]
    assert all(chunk.chunk_id.startswith("notes.pdf:p1:") for chunk in page_one)
    assert all(chunk.chunk_id.startswith("notes.pdf:p2:") for chunk in page_two)


def test_chunk_ids_are_stable_across_runs() -> None:
    document = _document("abcdefghijklmnopqrstuvwxyz", "second page text")

    first = chunk_document(document, chunk_size=10, overlap=2)
    second = chunk_document(document, chunk_size=10, overlap=2)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == make_chunk_id(Path("notes.pdf"), 1, 1)
    assert any(chunk.chunk_id == "notes.pdf:p2:c1" for chunk in first)


def test_invalid_chunk_parameters() -> None:
    document = _document("hello")

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(document, chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, chunk_size=10, overlap=-1)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_document(document, chunk_size=10, overlap=10)
