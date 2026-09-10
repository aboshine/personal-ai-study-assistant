# Personal AI Study Assistant

Local-first study assistant for learning from your own course materials.

This repository currently includes the Python project foundation, PDF text extraction, basic local chat via Ollama, PDF-aware question answering, chunking, local embeddings, a SQLite vector store, cosine similarity retrieval, a minimal local RAG pipeline, and multiple-choice quiz generation from retrieved chunks.

## Setup

Python 3.11+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

## Run

```powershell
python -m study_assistant
```

or:

```powershell
study-assistant
```

## PDF extraction

Extract page-numbered text from a local PDF (no RAG or chat):

```python
from study_assistant.pdf_extraction import extract_pdf

document = extract_pdf(r"C:\path\to\notes.pdf")
for page in document.pages:
    print(page.page_number, page.text)
```

Missing or invalid files raise `PdfNotFoundError` or `InvalidPdfError`.

## Local LLM chat (Ollama)

The LLM layer is provider-independent. The current provider is Ollama, using its local HTTP API. Provider, model, and base URL come from `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`). Embedding model settings are `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` (base URL is `LLM_BASE_URL`).

1. Install Ollama: https://ollama.com/download
2. Confirm the service is running (Windows installer usually starts it). Default API: `http://127.0.0.1:11434`
3. Pull the model named in `.env` (default `llama3.2`):

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Send a prompt from Python:

```python
from study_assistant.llm import complete

print(complete("Say hello in one sentence."))
```

Or from the CLI:

```powershell
python -m study_assistant "Say hello in one sentence."
```

Connection problems raise `LLMConnectionError`. Invalid HTTP/JSON responses raise `LLMResponseError`.

## Ask a question about a PDF

Extract the PDF, put page-numbered text in the prompt, and ask the configured LLM. The model is instructed to answer only from the document.

CLI:

```powershell
python -m study_assistant ask-pdf C:\path\to\notes.pdf "What is the definition of a stack?"
```

or:

```powershell
study-assistant ask-pdf C:\path\to\notes.pdf "What is the definition of a stack?"
```

Python:

```python
from study_assistant.pdf_qa import answer_from_pdf

print(answer_from_pdf(r"C:\path\to\notes.pdf", "What is the definition of a stack?"))
```

Requires Ollama running and the configured model pulled. This sends the full extracted text (not retrieval/RAG). Use `--help` for usage.

## Embeddings

Embeddings use a replaceable `EmbeddingClient`. The current implementation calls Ollama `POST /api/embed`.

```python
from study_assistant.chunking import chunk_document
from study_assistant.embeddings import embed_chunks, embed_text
from study_assistant.pdf_extraction import extract_pdf

vector = embed_text("Stacks use LIFO.")
chunks = chunk_document(extract_pdf(r"C:\path\to\notes.pdf"))
embedded = embed_chunks(chunks)
```

Pull the embedding model named in `.env` (default `nomic-embed-text`):

```powershell
ollama pull nomic-embed-text
```

Connection problems raise `EmbeddingConnectionError`. Invalid HTTP/JSON/vector responses raise `EmbeddingResponseError`.

## Vector store

Embedded chunks are stored in a local SQLite file. The path comes from `VECTOR_STORE_PATH` (default `DATA_DIR/vector_store.sqlite`). Similarity search ranks stored vectors with cosine similarity in Python. Pass an already-computed query embedding; the store does not call Ollama.

```python
from study_assistant.vector_store import get_vector_store

store = get_vector_store()
store.upsert(embedded)
loaded = store.get("notes.pdf:p1:c1")
for chunk in store.list_by_source(r"C:\path\to\notes.pdf"):
    print(chunk.chunk_id, chunk.page_number)
store.delete_by_source(r"C:\path\to\notes.pdf")
print(store.count())
store.close()
```

Similarity search (query vector must already be embedded):

```python
from study_assistant.vector_store import get_vector_store

store = get_vector_store()
matches = store.search(query_vector, top_k=5)
for match in matches:
    print(match.similarity, match.page_number, match.chunk_id, match.text)
store.close()
```

Or open a specific file:

```python
from study_assistant.vector_store import SqliteVectorStore

with SqliteVectorStore(r"C:\path\to\vector_store.sqlite") as store:
    store.upsert(embedded)
```

## RAG

Embed the question, retrieve the top-k stored chunks, and ask the LLM using only that context. The store still does not call Ollama; the embedding client creates the query vector.

```python
from study_assistant.rag import create_rag_pipeline

pipeline = create_rag_pipeline(top_k=5)
result = pipeline.answer("What is a stack?")
print(result.answer)
for index, source in enumerate(result.sources, start=1):
    print(f"[{index}] {source.source_path} page {source.page_number}")
```

Citation numbers in the answer match retrieved-chunk order: `[1]` is the first source, `[2]` the second, and so on. Example:

```
A stack is a LIFO structure. [1]
```

maps `[1]` to `result.sources[0]` (for example `notes.pdf`, page 1). The model is told to cite claims, never invent citation numbers, and say so if the context is insufficient. If nothing is retrieved, the pipeline returns a fixed “not enough information” answer and does not call the LLM.

## Quiz generation

Generate multiple-choice questions from retrieved chunks (not from a database query inside the quiz module). Each question has four options, one correct answer, an explanation, and source/page references.

```python
from study_assistant.llm import get_llm_client
from study_assistant.quiz import QuizGenerator

quiz = QuizGenerator(get_llm_client()).generate(
    retrieved_chunks, question_count=3, difficulty="hard"
)
for question in quiz.questions:
    print(question.question)
    for option in question.options:
        print(option.label, option.text)
    print("Correct:", question.correct_label)
    print(question.explanation)
    for source in question.sources:
        print(source.citation_index, source.source_path, source.page_number)
```

Pass chunks you already retrieved (for example `store.search(...)` or `RagResult.sources`). Empty context is rejected. Difficulty is `easy`, `medium` (default), or `hard`.

## Quiz evaluation

Scoring is deterministic and does not use an LLM. Each correct answer is 1 point; incorrect or unanswered questions are 0. Percentage is `correct / total × 100` (an empty quiz scores 0%). Question indexes are 0-based. Unlisted questions count as unanswered.

```python
from study_assistant.quiz_evaluation import SubmittedAnswer, evaluate_quiz

result = evaluate_quiz(
    quiz,
    (
        SubmittedAnswer(0, "A"),
        SubmittedAnswer(1, "C"),
    ),
)
print(result.correct_count, result.incorrect_count, result.unanswered_count, result.score_percent)
for item in result.question_results:
    print(item.question_index, item.selected_label, item.correct_label, item.is_correct)
    print(item.explanation)
```

## Tests

```powershell
pytest
```

