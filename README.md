# RAG Pipeline

A Retrieval-Augmented Generation system. It indexes a folder of documents, then answers questions using only the retrieved passages — with citations back to the source, instead of relying on the model's own memory.

## How it works

**Indexing** (run once): documents are loaded, split into overlapping chunks, converted into vector embeddings, and stored in a local vector database (ChromaDB).

**Querying** (run per question): the question is embedded with the same model, the closest matching chunks are retrieved, and an LLM generates an answer from that retrieved text, citing which passages it used.

Two ways to use it: a command-line interface, and a web app (FastAPI backend + browser frontend) with upload, delete, and reindex built in.

## Stack

Python, sentence-transformers, ChromaDB, Anthropic API, FastAPI, vanilla HTML/CSS/JS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your ANTHROPIC_API_KEY
```

## Usage — command line

```bash
# Index a folder of documents (.txt / .md / .pdf)
python ingest.py ./docs

# Ask a question
python query.py "your question"

# Check retrieval only, no API key needed
python query.py "your question" --no-llm

# Rebuild the index from scratch
python ingest.py ./docs --reset

# Evaluate retrieval quality against a test set
python eval.py
```

## Usage — web app

```bash
# Terminal 1 — the API
uvicorn api:app --reload --port 8000

# Terminal 2 — the frontend
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500/index.html` to ask questions, or `http://localhost:5500/upload.html` to upload, reindex, or delete documents.

## Configuration

Settings live in `config.py` (embedding model, chunk size, top_k, etc.) and can be overridden with environment variables.