from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


class TextChunker:
    """Splits text with awareness of paragraph, sentence, and word boundaries.

    Uses LangChain's RecursiveCharacterTextSplitter which tries each
    separator in order — paragraphs first, then newlines, then sentence
    delimiters, then spaces — so chunks rarely break mid-sentence.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    def chunk_text(self, text: str, source_path: str = "") -> List[Chunk]:
        splits = self.splitter.split_text(text)
        chunks: List[Chunk] = []
        actual_total = len(splits)

        for idx, split_text in enumerate(splits):
            if not split_text.strip():
                continue
            chunks.append(Chunk(
                text=split_text,
                metadata={
                    "source": source_path,
                    "chunk_index": idx,
                    "total_chunks": actual_total,
                },
            ))
        return chunks
