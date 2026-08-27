import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel

import config 
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.generator import generate_answer_stream
from rag.loader import SUPPORTED_EXTENSIONS, load_documents
from rag.retriever import Retriever
from rag.store import VectorStore

load_dotenv()

app = FastAPI(title="RAG API", description="API for RAG model", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = Path("docs")

_embedder = Embedder(config.EMBED_MODEL)
_store = VectorStore(config.PERSIST_DIR, config.COLLECTION_NAME)
_retriever = Retriever(Embedder(config.EMBED_MODEL), _store)

class QueryRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K

@app.get("/health")
def health() -> dict:
    return {"Status" : "Ok", "Chunks_indexed": _store.count()}


@app.get("/documents")
def list_documents() -> dict:
    DOCS_DIR.mkdir(exist_ok=True)
    chunk_counts = _store.chunk_counts_by_source()

    files = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            continue
        source = str(path.relative_to(DOCS_DIR))
        chunk_count = chunk_counts.get(source, 0)
        files.append({
            "filename": source,
            "size_bytes": path.stat().st_size,
            "indexed": chunk_count > 0,
            "chunks_indexed": chunk_count
        })

    return {"files": files, "total_chunks_indexed": _store.count()}


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    def event_stream():
        chunks = _retriever.retrieve(req.question, req.top_k)

        sources_payload = [
            {"index": i, "source": c.source, "distance": c.distance, "text": c.text}
            for i, c in enumerate(chunks, start=1)
        ]
        yield f"event: sources\ndata  {json.dumps(sources_payload)}\n\n"

        if not chunks: 
            yield f"event: error \ndata: {json.dumps({'message': 'No relevant chunks found.'})}\n\n"
            return 
        for piece in generate_answer_stream(req.question, chunks, config.LLM_MODEL):
            yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"

        yield f"event: done\ndata: {json.dumps({'message': 'Answer generation completed.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)) -> dict:

    DOCS_DIR.mkdir(exist_ok=True)

    saved_files = []
    rejected_files = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            rejected_files.append({"filename": upload.filename, "reason": f"unsupported type {suffix}"})
            continue

        dest = DOCS_DIR / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_files.append({"filename": upload.filename, "path": str(dest)})

    return {"saved": saved_files, "rejected": rejected_files}

@app.post("/reindex")
def reindex() -> dict:

    total_chunks = 0
    files_indexed = 0

    for doc in load_documents(DOCS_DIR):
        chunks = chunk_text(doc.text, doc.source, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue 

        embeddings = _embedder.embed([c.text for c in chunks])
        _store.add(
            ids=[f"{c.source}::{c.index}"for c in chunks],
            texts = [c.text for c in chunks],
            embeddings = embeddings,
            metadatas = [{"source": c.source, "index": c.index} for c in chunks]
        )

        total_chunks += len(chunks)
        files_indexed += 1

    return {"files_indexed": files_indexed, "total_chunks_indexed": total_chunks, 
            "Collection_size": _store.count()}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)