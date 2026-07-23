# Data ingestion script

import argparse
import config 
from rag.chunker import chunk_text
from rag.embedder import Embedder 
from rag.loader import load_documents
from rag.store import VectorStore

def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents for RAG.")
    parser.add_argument("source_dir", help="Folder of .txt/.md/.pdf files to index")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing index before re-building")
    args = parser.parse_args()

    embedder = Embedder(config.EMBED_MODEL)
    store = VectorStore(config.PERSIST_DIR, config.COLLECTION_NAME)    
    if args.reset:
        store.reset()
        print("Cleared Existing index.")

    total_chunks = 0
    for doc in load_documents(args.source_dir):
        chunks = chunk_text(doc.text, doc.source, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue 

        embeddings = embedder.embed ([c.text for c in chunks])
        store.add(
            ids=[f"{c.source}::{c.index}" for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )
        total_chunks += len(chunks)
        print (f" indexed {len(chunks):>4} chunks from {doc.source}")

    print(f"\nDone. {total_chunks} chunks indexed; collection now holds {store.count()} chunks.")

if __name__ == "__main__":
    main()