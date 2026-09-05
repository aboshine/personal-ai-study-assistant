# Personal AI Study Assistant — Project Specification

## 1. Project Goal

Build a local-first AI Study Assistant that helps a university student learn from their own course materials.

The system must eventually support:

- PDF/lecture material ingestion
- course-specific AI tutoring
- RAG-based question answering
- source/page citations
- quiz generation and evaluation
- student knowledge tracking
- adaptive quizzes
- spaced repetition
- lab preparation
- personalized study planning
- answer verification and evaluation

The application is a serious portfolio/software-engineering project, not a simple chatbot wrapper.

## 2. Development Principles

- Build incrementally.
- Keep the architecture modular.
- Do not implement future features before they are required.
- Prefer simple solutions over unnecessary complexity.
- Keep components replaceable.
- Do not hard-code model providers.
- Local-first development.
- Do not require paid APIs.
- Write readable, maintainable Python.
- Use environment variables for configuration and secrets.
- Never commit secrets or personal study materials to Git.

## 3. AI Architecture

The LLM layer must be provider-independent.

Initial provider:

- Ollama
- Local quantized LLM

The architecture must allow a future cloud/API provider to replace Ollama without rewriting the application.

## 4. Course Isolation

Each course must have isolated:

- materials
- conversations
- quizzes
- knowledge/progress data

Information from one course must not accidentally enter another course's context.

## 5. RAG Requirements

When RAG is implemented:

PDF → extraction → chunking → embeddings → vector storage → retrieval → LLM

Course/Strict mode must:

- prioritize uploaded course materials;
- answer from retrieved material;
- provide source/page information when possible;
- avoid unsupported claims;
- explicitly state when the material does not contain enough information.

## 6. Verification

The system should eventually verify AI-generated content where practical.

For numerical/math problems:

LLM → generate → deterministic verification → explanation

For generated quizzes:

source material → question/answer generation → source verification → accept/regenerate

Never assume that an LLM response is automatically correct.

## 7. Code Quality

Use:

- type hints where useful;
- clear module boundaries;
- small functions;
- meaningful names;
- error handling;
- logging where appropriate;
- tests for important logic.

Avoid:

- unnecessary abstractions;
- giant files;
- duplicated logic;
- hidden global state;
- hard-coded paths;
- hard-coded API/model configuration.

## 8. Development Roadmap

Build in this order:

V1 — Project foundation + PDF text extraction + basic local chat

V2 — RAG

V3 — Source/page citations

V4 — Quiz generation

V5 — Quiz evaluation and result storage

V6 — Knowledge profile

V7 — Adaptive quizzes

V8 — Spaced repetition

V9 — Multiple courses

V10 — Personalized study planner

V11 — Evaluation and verification

Do not skip ahead unless explicitly instructed.

## 9. Current Development Rule

At every stage:

1. Implement only the requested feature.
2. Run and test it.
3. Fix errors.
4. Keep the project runnable.
5. Do not implement future roadmap items.
6. Explain changes briefly after implementation.

The human developer will manually inspect and understand the generated code.

## 10. Git

Use small, meaningful commits.

Commit after a stable milestone.

Do not commit:

- `.env`
- secrets
- virtual environments
- large personal PDFs
- generated private study data
- model files

## 11. Current Task

Only prepare the initial Python project foundation.

Do NOT implement:

- RAG
- embeddings
- vector database
- quiz generation
- knowledge tracking
- adaptive learning
- frontend
- agents

Keep the foundation minimal and ready for future expansion.