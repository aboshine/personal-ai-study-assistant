# Personal AI Study Assistant

Local-first study assistant for learning from your own course materials.

This repository currently includes the Python project foundation, PDF text extraction, basic local chat via Ollama, and PDF-aware question answering (full document in the prompt; not RAG).

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

The LLM layer is provider-independent. The current provider is Ollama, using its local HTTP API. Provider, model, and base URL come from `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`).

1. Install Ollama: https://ollama.com/download
2. Confirm the service is running (Windows installer usually starts it). Default API: `http://127.0.0.1:11434`
3. Pull the model named in `.env` (default `llama3.2`):

```powershell
ollama pull llama3.2
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

## Tests

```powershell
pytest
```

