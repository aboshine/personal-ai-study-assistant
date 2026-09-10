"""Persistent local vector store backed by SQLite.

Stores embedded chunks and ranks them with local cosine similarity.
This module does not generate embeddings or call an embedding provider.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from study_assistant.config import Settings, load_settings
from study_assistant.embeddings import EmbeddedChunk

DEFAULT_TOP_K = 5

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL
)
"""

_CREATE_SOURCE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks (source_path)
"""


class VectorStoreError(Exception):
    """Raised when stored embedding data cannot be read, written, or compared."""


@dataclass(frozen=True)
class SimilarChunk:
    """A stored chunk ranked against a query embedding."""

    chunk_id: str
    source_path: Path
    page_number: int
    text: str
    similarity: float


class VectorStore(ABC):
    """Storage interface. Independent of how embeddings were produced."""

    @abstractmethod
    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert chunks or replace any row with the same chunk_id."""

    @abstractmethod
    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        """Return one chunk by id, or None if it is not stored."""

    @abstractmethod
    def list_by_source(self, source_path: str | Path) -> tuple[EmbeddedChunk, ...]:
        """Return all chunks for a source document, ordered by page then id."""

    @abstractmethod
    def delete_by_source(self, source_path: str | Path) -> int:
        """Delete all chunks for a source document. Returns the number of rows removed."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[SimilarChunk, ...]:
        """Return the top-k stored chunks by cosine similarity to `query_vector`."""


class SqliteVectorStore(VectorStore):
    """SQLite-backed vector store. Creates the schema on first use."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        rows = [
            (
                chunk.chunk_id,
                _normalize_source(chunk.source_path),
                chunk.page_number,
                chunk.text,
                serialize_vector(chunk.vector),
            )
            for chunk in chunks
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO chunks (chunk_id, source_path, page_number, chunk_text, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    page_number = excluded.page_number,
                    chunk_text = excluded.chunk_text,
                    embedding = excluded.embedding
                """,
                rows,
            )

    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        row = self._connection.execute(
            "SELECT chunk_id, source_path, page_number, chunk_text, embedding FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_chunk(row)

    def list_by_source(self, source_path: str | Path) -> tuple[EmbeddedChunk, ...]:
        rows = self._connection.execute(
            """
            SELECT chunk_id, source_path, page_number, chunk_text, embedding
            FROM chunks
            WHERE source_path = ?
            ORDER BY page_number ASC, chunk_id ASC
            """,
            (_normalize_source(source_path),),
        ).fetchall()
        return tuple(_row_to_chunk(row) for row in rows)

    def delete_by_source(self, source_path: str | Path) -> int:
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM chunks WHERE source_path = ?",
                (_normalize_source(source_path),),
            )
        return cursor.rowcount

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row is not None else 0

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[SimilarChunk, ...]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = _as_query_vector(query_vector)

        rows = self._connection.execute(
            "SELECT chunk_id, source_path, page_number, chunk_text, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return ()

        ranked: list[SimilarChunk] = []
        for row in rows:
            stored = deserialize_vector(row["embedding"])
            ranked.append(
                SimilarChunk(
                    chunk_id=row["chunk_id"],
                    source_path=Path(row["source_path"]),
                    page_number=int(row["page_number"]),
                    text=row["chunk_text"],
                    similarity=cosine_similarity(query, stored),
                )
            )

        ranked.sort(key=lambda item: (-item.similarity, item.chunk_id))
        return tuple(ranked[:top_k])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteVectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self._connection:
            self._connection.execute(_CREATE_CHUNKS_TABLE)
            self._connection.execute(_CREATE_SOURCE_INDEX)


def get_vector_store(settings: Settings | None = None) -> SqliteVectorStore:
    """Open the SQLite store at the configured path, creating it if needed."""
    resolved = settings if settings is not None else load_settings()
    return SqliteVectorStore(resolved.vector_store_path)


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Encode a vector as little-endian IEEE-754 float64 values."""
    if not vector:
        raise VectorStoreError("Cannot store an empty embedding vector")
    values = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VectorStoreError("Embedding vector contains a non-numeric value")
        values.append(float(item))
    return struct.pack("<" + "d" * len(values), *values)


def deserialize_vector(blob: bytes) -> tuple[float, ...]:
    """Decode little-endian IEEE-754 float64 bytes into a vector."""
    if not blob or len(blob) % 8 != 0:
        raise VectorStoreError("Stored embedding blob is missing or truncated")
    count = len(blob) // 8
    return struct.unpack("<" + "d" * count, blob)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity of two vectors. Dimensions must match."""
    a = _numeric_vector(left, label="Query embedding")
    b = _numeric_vector(right, label="Stored embedding")
    if len(a) != len(b):
        raise VectorStoreError(
            f"Embedding dimension mismatch: query has {len(a)} values, stored has {len(b)}"
        )

    left_norm = math.sqrt(sum(value * value for value in a))
    right_norm = math.sqrt(sum(value * value for value in b))
    if left_norm == 0.0:
        raise VectorStoreError("Query embedding has zero magnitude")
    if right_norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (left_norm * right_norm)


def _as_query_vector(vector: Sequence[float]) -> tuple[float, ...]:
    return _numeric_vector(vector, label="Query embedding")


def _numeric_vector(vector: Sequence[float], *, label: str) -> tuple[float, ...]:
    if not vector:
        raise VectorStoreError(f"{label} must not be empty")
    values: list[float] = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VectorStoreError(f"{label} contains a non-numeric value")
        values.append(float(item))
    return tuple(values)


def _normalize_source(source_path: str | Path) -> str:
    return Path(source_path).as_posix()


def _row_to_chunk(row: sqlite3.Row) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=row["chunk_id"],
        text=row["chunk_text"],
        source_path=Path(row["source_path"]),
        page_number=int(row["page_number"]),
        vector=deserialize_vector(row["embedding"]),
    )
