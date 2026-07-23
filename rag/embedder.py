# Embedding generation module
"""
Turns text into vectors positioned so that semantically similar text lands nearby.
the same model must embed both documents and questions at query time so that the shared 
space is what makes meaning based search work.
"""

from sentence_transformers import SentenceTransformer

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