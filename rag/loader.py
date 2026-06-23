# Document loader module
from dataclasses import dataclass 
from pathlib import Path
from typing import Iterator 
from pypdf import PdfReader

SUPPORTED_EXTENSIONS  = [".txt",".pdf",".md"]

@dataclass
class Document:
    source  : str   #relative path used for citations
    text    : str

def load_documents(source_dir: str) -> Iterator[Document]:
# Goes through a folder and all subfolders, finds every file that is supported and ceates
# a Document object for each one, yielding it to the caller.
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = _extract(path)
        if text.strip():
            yield Document(source=str(path.relative_to(root)), text=text)

def _extract(path: Path) -> str:
    # Extracts text from a file, handling different formats based on the file extension.
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    # Reads a PDF file and extracts text from all its pages, concatenating them into a single string.
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)