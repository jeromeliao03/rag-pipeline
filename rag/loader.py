#Pull clean text out of source files.
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pdfplumber

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Matches a lowercase letter immediately followed by an uppercase letter —
# e.g. the boundary inside "PromptManagement" or "HallucinationDetection".
# This is the most common shape glued-together PDF words take, so inserting
# a space at each match recovers a lot of readability regardless of what
# caused the original extraction to lose the space.
_GLUED_WORD_BOUNDARY = re.compile(r"([a-z])([A-Z])")


@dataclass
class Document:
    source: str  # relative path, used later for citations
    text: str


def load_documents(source_dir: str) -> Iterator[Document]:
    """Walk a directory recursively and yield one Document per supported file."""
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = _extract(path)
        if text.strip():
            yield Document(source=str(path.relative_to(root)), text=text)


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2)
            pages.append(" ".join(w["text"] for w in words))
    text = "\n".join(pages)
    return _fix_glued_words(text)   # ← route through the cleanup step


def _fix_glued_words(text: str) -> str:
    """Insert a space wherever a lowercase letter is immediately followed by
    an uppercase one — a safety net for words the PDF extractor still glued
    together (e.g. slides with structurally overlapping text layers that no
    extraction tolerance can fully separate).
    """
    return _GLUED_WORD_BOUNDARY.sub(r"\1 \2", text)