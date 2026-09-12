# Personal AI Study Assistant

Local-first study assistant for your own lecture PDFs. It indexes materials on your machine, answers questions with page citations, runs quizzes, and builds a study plan from your results. No paid API is required.

## 5-minute demo

You need Python 3.11+ and [Ollama](https://ollama.com/download).

### 1. Install the app

Windows (PowerShell):

```powershell
git clone <this-repo-url>
cd personal-ai-study-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
```

macOS / Linux:

```bash
git clone <this-repo-url>
cd personal-ai-study-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

### 2. Start Ollama and pull models

Install Ollama, then:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Leave Ollama running. Default API: `http://127.0.0.1:11434`.

### 3. Start the web app

```powershell
python -m web
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 4. Study

1. **Library** — upload a course PDF. Indexing can take a minute.
2. **Ask** — question the notes; answers include source and page citations.
3. **Quiz** — generate a multiple-choice quiz (or Adaptive, which uses History).
4. **History** / **Plan** — scores and what to review next.

If Ask or Quiz says nothing is indexed, go back to Library.

## Configuration

Copy `.env.example` to `.env`. Models and paths:

- `LLM_MODEL` (default `llama3.2`)
- `EMBEDDING_MODEL` (default `nomic-embed-text`)
- `LLM_BASE_URL` (default `http://127.0.0.1:11434`)
- `DATA_DIR` (local SQLite stores and uploaded PDFs)

## CLI (optional)

```powershell
python -m study_assistant index C:\path\to\notes.pdf
python -m study_assistant ask "What is a stack?"
python -m study_assistant quiz stacks --count 2 --difficulty medium
python -m study_assistant plan
```

`python -m study_assistant --help` lists commands. The web app is the intended demo path.

## Tests

```powershell
pip install -e ".[dev]"
pytest
```
