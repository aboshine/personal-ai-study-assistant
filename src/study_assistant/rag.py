"""Minimal RAG: embed question, retrieve chunks, ask the LLM from that context.

This module does not implement embeddings, similarity, or persistence.
It wires existing clients and the vector store together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from study_assistant.config import Settings
from study_assistant.embeddings import EmbeddingClient, get_embedding_client
from study_assistant.llm import LLMClient, get_llm_client
from study_assistant.vector_store import (
    DEFAULT_TOP_K,
    SimilarChunk,
    VectorStore,
    get_vector_store,
)

NO_CONTEXT_ANSWER = (
    "The retrieved study materials do not contain enough information to answer this question."
)


@dataclass(frozen=True)
class RagResult:
    """Answer plus the retrieved chunks used as context."""

    answer: str
    sources: tuple[SimilarChunk, ...]


class RagPipeline:
    """Question → query embedding → retrieval → grounded LLM answer."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingClient,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.embeddings = embeddings
        self.store = store
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str) -> RagResult:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question must not be empty")

        query_vector = self.embeddings.embed(cleaned)
        sources = self.store.search(query_vector, top_k=self.top_k)
        if not sources:
            return RagResult(answer=NO_CONTEXT_ANSWER, sources=())

        prompt = build_rag_prompt(cleaned, sources)
        return RagResult(answer=self.llm.complete(prompt), sources=sources)


def create_rag_pipeline(
    settings: Settings | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> RagPipeline:
    """Build a pipeline from configured embedding, store, and LLM clients."""
    return RagPipeline(
        embeddings=get_embedding_client(settings),
        store=get_vector_store(settings),
        llm=get_llm_client(settings),
        top_k=top_k,
    )


def build_rag_prompt(question: str, chunks: Sequence[SimilarChunk]) -> str:
    """Build a deterministic prompt from retrieved chunks. Does not invent context."""
    if not chunks:
        raise ValueError("Cannot build a RAG prompt without retrieved chunks")

    source_list = "\n".join(
        f"[{index}] {chunk.source_path.as_posix()}, page {chunk.page_number}"
        for index, chunk in enumerate(chunks, start=1)
    )
    context_blocks = "\n\n".join(
        _format_retrieved_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)
    )
    max_citation = len(chunks)
    return (
        "You are a study assistant.\n"
        "Answer the user's question using ONLY the retrieved context below.\n"
        "If the context does not contain enough information to answer, say so clearly.\n"
        "Do not use outside knowledge.\n"
        "Do not invent facts that are not supported by the context.\n"
        "Cite factual claims with [1], [2], etc. matching the numbered sources below, "
        "in the same order as the retrieved chunks.\n"
        "Place a citation immediately after each factual claim it supports.\n"
        f"Use only citation numbers from [1] to [{max_citation}]. "
        "Never invent citations or use numbers that are not listed.\n"
        "Do not invent sources or page numbers.\n"
        "\n"
        "Sources:\n"
        f"{source_list}\n"
        "\n"
        "Retrieved context:\n"
        f"{context_blocks}\n"
        "\n"
        "Question:\n"
        f"{question.strip()}\n"
    )


def _format_retrieved_chunk(index: int, chunk: SimilarChunk) -> str:
    body = chunk.text if chunk.text.strip() else "(no text extracted from this chunk)"
    return (
        f"[{index}] source={chunk.source_path.as_posix()} page={chunk.page_number} "
        f"chunk_id={chunk.chunk_id} similarity={chunk.similarity:.6f}\n"
        f"{body}"
    )
