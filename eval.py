import argparse
import json

import config
from rag.embedder import Embedder
from rag.eval import EvalQuestion, evaluate_retrieval
from rag.retriever import Retriever
from rag.store import VectorStore


def load_questions(path: str) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalQuestion(question=q["question"], expected_source=q["expected_source"]) for q in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a known test set.")
    parser.add_argument("--questions", default="eval_questions.json", help="Path to the test set JSON file")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    store = VectorStore(config.PERSIST_DIR, config.COLLECTION_NAME)
    if store.count() == 0:
        print("The index is empty. Run ingest.py first.")
        return

    questions = load_questions(args.questions)
    retriever = Retriever(Embedder(config.EMBED_MODEL), store)
    report = evaluate_retrieval(questions, retriever, args.top_k)

    print(f"\nRetrieval evaluation — {len(questions)} questions, top_k={args.top_k}")
    print("-" * 60)
    for r in report.results:
        status = f"HIT  (rank {r.rank})" if r.hit else "MISS"
        print(f"[{status:<14}] {r.question}")
        if not r.hit:
            print(f"    expected: {r.expected_source}")
            print(f"    got:      {r.retrieved_sources}")

    print("-" * 60)
    print(f"Hit rate: {report.hit_rate:.2%}")
    print(f"MRR:      {report.mrr:.3f}")


if __name__ == "__main__":
    main()