"""Chunking utilities for splitting a document into smaller retrievable text blocks.

This module turns one long source file into a list of chunks, each with its
source filename and chunk number so later retrieval can cite the right passage.
"""

from dataclasses import dataclass

# Chunk model
@dataclass
class Chunk:
    text: str
    source: str
    index: int  # position of the chunk within its source document


# Chunk text into overlapping windows
def chunk_text(
        text: str,
        source: str,
        size_words: int,
        overlap_words: int
) -> list[Chunk]:
    if size_words <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if overlap_words < 0:
        raise ValueError("Overlap size must be non-negative")

    words = text.split()
    if not words:
        return []

    step = size_words - overlap_words
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start:start + size_words]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), source=source, index=len(chunks)))
        if start + size_words >= len(words):
            break
    return chunks