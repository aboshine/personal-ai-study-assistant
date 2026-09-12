from pathlib import Path

import pytest

from study_assistant.embeddings import EmbeddedChunk
from study_assistant.vector_store import (
    SimilarChunk,
    SqliteVectorStore,
    VectorStoreError,
    cosine_similarity,
    deserialize_vector,
    serialize_vector,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "notes.pdf",
    page: int = 1,
    vector: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        source_path=Path(source),
        page_number=page,
        vector=vector,
    )


def test_serialize_vector_is_deterministic() -> None:
    vector = (0.1, -2.5, 3.0)
    assert serialize_vector(vector) == serialize_vector(vector)
    assert deserialize_vector(serialize_vector(vector)) == vector


def test_serialize_vector_rejects_empty() -> None:
    with pytest.raises(VectorStoreError, match="empty"):
        serialize_vector(())


def test_upsert_get_and_count(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    chunk = _chunk("notes.pdf:p1:c1", "Stacks use LIFO.")

    with SqliteVectorStore(db_path) as store:
        store.upsert([chunk])
        loaded = store.get("notes.pdf:p1:c1")
        assert loaded == chunk
        assert store.count() == 1
        assert store.get("missing") is None


def test_upsert_replaces_existing_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    original = _chunk("notes.pdf:p1:c1", "old text", vector=(1.0, 0.0))
    updated = _chunk("notes.pdf:p1:c1", "new text", page=2, vector=(0.0, 1.0))

    with SqliteVectorStore(db_path) as store:
        store.upsert([original])
        store.upsert([updated])
        loaded = store.get("notes.pdf:p1:c1")
        assert loaded is not None
        assert loaded.text == "new text"
        assert loaded.page_number == 2
        assert loaded.vector == (0.0, 1.0)
        assert store.count() == 1


def test_list_and_delete_by_source(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    notes = [
        _chunk("notes.pdf:p1:c1", "first", source="notes.pdf", page=1),
        _chunk("notes.pdf:p2:c1", "second", source="notes.pdf", page=2),
    ]
    other = _chunk("lab.pdf:p1:c1", "other", source="lab.pdf")

    with SqliteVectorStore(db_path) as store:
        store.upsert([*notes, other])
        listed = store.list_by_source("notes.pdf")
        assert [chunk.chunk_id for chunk in listed] == ["notes.pdf:p1:c1", "notes.pdf:p2:c1"]
        deleted = store.delete_by_source(Path("notes.pdf"))
        assert deleted == 2
        assert store.list_by_source("notes.pdf") == ()
        remaining = store.get("lab.pdf:p1:c1")
        assert remaining is not None
        assert remaining.text == "other"
        assert store.count() == 1
        assert store.list_source_counts() == ((Path("lab.pdf"), 1),)


def test_persists_across_separate_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "store.sqlite"
    chunk = _chunk("notes.pdf:p1:c1", "Persisted text", vector=(0.25, 0.5, 0.75))

    store = SqliteVectorStore(db_path)
    store.upsert([chunk])
    store.close()

    reopened = SqliteVectorStore(db_path)
    loaded = reopened.get("notes.pdf:p1:c1")
    assert loaded == chunk
    assert reopened.count() == 1
    reopened.close()


def test_cosine_similarity_of_aligned_and_orthogonal_vectors() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_search_ranks_by_descending_cosine_similarity(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    chunks = [
        _chunk("notes.pdf:p1:c1", "exact", page=1, vector=(1.0, 0.0, 0.0)),
        _chunk("notes.pdf:p2:c1", "partial", page=2, vector=(0.6, 0.8, 0.0)),
        _chunk("notes.pdf:p3:c1", "orthogonal", page=3, vector=(0.0, 1.0, 0.0)),
    ]

    with SqliteVectorStore(db_path) as store:
        store.upsert(chunks)
        results = store.search((1.0, 0.0, 0.0), top_k=3)

    assert [item.chunk_id for item in results] == [
        "notes.pdf:p1:c1",
        "notes.pdf:p2:c1",
        "notes.pdf:p3:c1",
    ]
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity == pytest.approx(0.6)
    assert results[2].similarity == pytest.approx(0.0)
    assert results[0].source_path == Path("notes.pdf")
    assert results[0].page_number == 1
    assert results[0].text == "exact"
    assert isinstance(results[0], SimilarChunk)


def test_search_respects_top_k(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    chunks = [
        _chunk("a", "a", vector=(1.0, 0.0)),
        _chunk("b", "b", vector=(0.8, 0.2)),
        _chunk("c", "c", vector=(0.0, 1.0)),
    ]

    with SqliteVectorStore(db_path) as store:
        store.upsert(chunks)
        results = store.search((1.0, 0.0), top_k=2)

    assert len(results) == 2
    assert [item.chunk_id for item in results] == ["a", "b"]


def test_search_empty_store_returns_no_results(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"

    with SqliteVectorStore(db_path) as store:
        assert store.search((1.0, 0.0, 0.0), top_k=5) == ()


def test_search_rejects_incompatible_dimensions(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    chunk = _chunk("notes.pdf:p1:c1", "three dims", vector=(1.0, 0.0, 0.0))

    with SqliteVectorStore(db_path) as store:
        store.upsert([chunk])
        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            store.search((1.0, 0.0), top_k=1)
