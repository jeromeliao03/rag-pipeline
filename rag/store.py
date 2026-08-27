import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        # upsert = replace if the ID exists, insert if it doesn't.
        # Makes re-running ingest.py on the same files safe — no duplicates.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        return self.collection.count()

    def chunk_counts_by_source(self) -> dict[str, int]:

        if self.count() == 0:
            return {}

        result = self.collection.get(include=['metadatas'])
        counts: dict[str, int] = {}
        for meta in result['metadatas']:
            source = meta.get('source', 'unknown')
            counts[source] = counts.get(source, 0) + 1
        return counts

    def reset(self) -> None:
        """Wipe the collection for a clean rebuild."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)