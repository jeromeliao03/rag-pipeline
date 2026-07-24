# Document retrieval module

from dataclasses import dataclass

from .embedder import Embedder
from .store import VectorStore

@dataclass
class RetrievedChunk:
    text: str
    source: str
    index: int 
    distance: float #lower = more similar 

class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        # Embed with the same Model used at index time 
        # anything within one shared vector space is comparable, so the same model 
        # must be used for both indexing and retrieval.

        query_vector = self.embedder.embed([question])[0]
        result = self.store.query(query_vector, top_k)

        #Chroma nests every field one level deeeper than expected 
        #Because API supports batching several queries at once

        documents = result['documents'][0]
        metadatas = result['metadatas'][0]
        distances = result['distances'][0]

        #Zip the three parallel lists back into one object per chunk
        return [
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                index=int(meta.get("index", -1)),
                distance=float(dist),
            )
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]