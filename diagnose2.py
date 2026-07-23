from pathlib import Path
from rag.loader import _read_pdf

path = Path(r"docs\Ai_Engineering\Seminar 3 - Software Engineering & DevOps AI.pdf")
text = _read_pdf(path)

idx = text.find("PromptManagement")
if idx == -1:
    print("NOT glued — fix worked on this file.")
else:
    print(f"STILL GLUED at position {idx}:")
    print(repr(text[max(0, idx - 30):idx + 60]))