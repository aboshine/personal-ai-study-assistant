# Personal AI Study Assistant

Local-first study assistant for learning from your own course materials.

This repository currently includes a local CLI, a minimal Flask Library UI for PDF indexing, PDF text extraction, indexing into the local vector store, basic local chat via Ollama, PDF-aware question answering, chunking, local embeddings, a SQLite vector store, cosine similarity retrieval, a minimal local RAG pipeline, multiple-choice quiz generation from retrieved chunks, quiz evaluation, persistent quiz attempt history, and topic-level knowledge tracking.

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
python -m study_assistant index C:\path\to\notes.pdf
python -m study_assistant ask "What is a stack?"
python -m study_assistant quiz stacks --count 2 --difficulty medium
python -m study_assistant adaptive stacks --topics Queues,Stacks
python -m study_assistant plan
python -m study_assistant evaluate quiz.json A B
python -m web
```

or:

```powershell
study-assistant
study-assistant-web
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

## Index a PDF

Extract, chunk, embed, and store a local PDF in the vector store. Re-indexing the same file replaces its previous chunks.

```powershell
python -m study_assistant index C:\path\to\notes.pdf
```

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

Generate multiple-choice questions from retrieved chunks (not from a database query inside the quiz module). Each question has four options, one correct answer, an explanation, optional topic metadata, and source/page references.

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

`python -m study_assistant evaluate quiz.json A B` scores a saved quiz JSON file and records the attempt in local SQLite history (`QUIZ_ATTEMPT_STORE_PATH`, default `DATA_DIR/quiz_attempts.sqlite`). Adaptive quizzes and `plan` read that history. The store does not score quizzes or call Ollama.

```python
from study_assistant.quiz_attempt_store import SqliteQuizAttemptStore, get_quiz_attempt_store
from study_assistant.quiz_attempts import create_quiz_attempt

attempt = create_quiz_attempt(quiz, result)
with SqliteQuizAttemptStore(r"C:\path\to\quiz_attempts.sqlite") as store:
    store.add(attempt)
    loaded = store.get(attempt.attempt_id)
```

Or open the configured path with `get_quiz_attempt_store()`.

## Topic knowledge tracking

Topic stats are computed from completed attempts only. Unanswered questions count toward a topic appearing in an attempt but not toward accuracy. Questions without a topic are ignored.

```python
from study_assistant.topic_tracking import summarize_topic_performance

for item in summarize_topic_performance(store.list_all()):
    print(item.topic, item.attempt_count, item.questions_answered, item.correct_count, item.accuracy)
```

## Adaptive quizzes

Topic choice and difficulty come from stored topic stats, not from the LLM. `generate_adaptive_quiz` passes that plan into `QuizGenerator`, so the prompt includes the selected topics and difficulty. Weaker topics get more questions. With no history the default is `medium` and the usual question count; pass `available_topics` to start from those names (treated as unpracticed).

```python
from study_assistant.adaptive_quiz import generate_adaptive_quiz, plan_adaptive_quiz

plan = plan_adaptive_quiz(summarize_topic_performance(store.list_all()), question_count=3)
quiz = generate_adaptive_quiz(
    retrieved_chunks,
    llm=get_llm_client(),
    attempts=store.list_all(),
    question_count=3,
)
```

## Tests

```powershell
pytest
```

