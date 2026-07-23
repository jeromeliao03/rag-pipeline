# Query script for RAG pipeline
"""Query entry point.

  python query.py "What is the refund policy?"
  python query.py "..." --top-k 6
  python query.py "..." --no-llm

Runs the online half of the pipeline: retrieve -> (rerank) -> generate.
"""
import argparse
from dotenv import load_dotenv

import config
from rag.embedder import Embedder
from rag.generator import generate_answer
from rag.retriever import Retriever
from rag.store import VectorStore

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the index.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--no-llm", action="store_true", help="Show retrieved passages only, skip the generation step")
    args = parser.parse_args()

    store = VectorStore(config.PERSIST_DIR, config.COLLECTION_NAME)
    if store.count() == 0:
        print("The index is empty. Run ingest.py first.")
        return

    retriever = Retriever(Embedder(config.EMBED_MODEL), store)
    chunks = retriever.retrieve(args.question, args.top_k)

    print("\nRetrieved passages")
    print("-" * 60)
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text[:160].replace("\n", " ")
        print(f"[{i}] {chunk.source}  (distance {chunk.distance:.3f})")
        print(f"    {preview}...\n")

    if args.no_llm:
        return

    print("Answer")
    print("-" * 60)
    answer = generate_answer(args.question, chunks, config.LLM_MODEL)
    print(answer)


if __name__ == "__main__":
    main()