"""Evaluation helpers for checking how well retrieval finds the expected source.

This file defines small evaluation objects and runs a simple retrieval test to
measure hit rate and MRR for a set of questions against the document store.
"""

from dataclasses import dataclass, field
from .retriever import Retriever

# Evaluation models
@dataclass
class EvalQuestion:
    question: str
    expected_source: str  # source filename that should appear in the retriever results


@dataclass
class QuestionResult:
    question: str
    expected_source: str
    retrieved_source: list[str]
    hit: bool
    rank: int | None  # 1-based position of the first matching source or None


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        reciprocal_ranks = [1 / r.rank if r.rank else 0.0 for r in self.results]
        return sum(reciprocal_ranks) / len(reciprocal_ranks)


# Retrieval evaluation
def evaluate_retrieval(
    questions: list[EvalQuestion],
    retriever: Retriever,
    top_k: int
) -> EvalReport:

    report = EvalReport()
    for eq in questions:
        chunks = retriever.retrieve(eq.question, top_k)
        retrieved_sources = [c.source for c in chunks]

        rank = None
        for i, source in enumerate(retrieved_sources, start=1):
            if eq.expected_source.lower() in source.lower():
                rank = i
                break

        report.results.append(QuestionResult(
            question=eq.question,
            expected_source=eq.expected_source,
            retrieved_source=retrieved_sources,
            hit=rank is not None,
            rank=rank,
        ))
    return report