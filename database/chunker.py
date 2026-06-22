from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_path: str = "") -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("chunk_size must be greater than chunk_overlap")

        total = (len(text) + step - 1) // step
        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if not chunk_text.strip():
                continue
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "source": source_path,
                    "chunk_index": len(chunks),
                    "total_chunks": total,
                }
            ))

        actual_total = len(chunks)
        for c in chunks:
            c.metadata["total_chunks"] = actual_total

        return chunks
