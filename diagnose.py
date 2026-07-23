import chromadb

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection(name="documents")

# Pull everything back and search for the specific glued phrase
results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if "prompt management" in doc.lower() or "promptmanagement" in doc.lower():
        print(f"--- {meta['source']} chunk {meta['index']} ---")
        print(doc)
        print()