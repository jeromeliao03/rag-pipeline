# Configuration settings for RAG pipeline
import os

# --- Models -----------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "200"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Storage ----------------------------------------------------------------
PERSIST_DIR = os.getenv("PERSIST_DIR", ".chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")