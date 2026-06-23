# Text chunking module
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    index: int #position of the chunk within its source document 

