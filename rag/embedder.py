"""Embedding helpers for turning text into vector representations.

This file loads a sentence embedding model and converts chunks or questions into
numeric vectors so similar content can be matched by distance in vector space.
"""

from sentence_transformers import SentenceTransformer

# Embedding wrapper
class Embedder:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()