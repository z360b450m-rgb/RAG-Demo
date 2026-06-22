import numpy as np
from pathlib import Path
from typing import Dict, Generator, List

from app_config import CHUNK_OVERLAP, CHUNK_SIZE, MMR_FETCH_K, MMR_LAMBDA, TOP_K
from core.llm import EmbeddingService, build_llm
from core.prompt import DIRECT_RAG_SYSTEM_PROMPT
from database.chunker import TextChunker
from database.document_loader import load_document
from database.vector_store import SearchResult, VectorStore


def mmr_rerank(
    query_embedding: List[float],
    results: List[SearchResult],
    result_embeddings: List[List[float]],
    top_k: int,
    lambda_param: float,
) -> List[SearchResult]:
    """Maximum Marginal Relevance re-ranking: balance relevance and diversity."""
    if len(results) <= top_k:
        return results

    selected: List[int] = []
    remaining = list(range(len(results)))

    best_idx = min(remaining, key=lambda i: results[i].score)
    selected.append(best_idx)
    remaining.remove(best_idx)

    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_idx = remaining[0]

        for idx in remaining:
            sim_query = 1.0 - results[idx].score
            sim_selected = 0.0
            for s in selected:
                cos_sim = np.dot(result_embeddings[idx], result_embeddings[s])
                sim_selected = max(sim_selected, cos_sim)

            mmr_score = lambda_param * sim_query - (1.0 - lambda_param) * sim_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [results[i] for i in selected]


class DirectRAGPipeline:
    """Always-retrieve RAG pipeline (non-agent mode)."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm = build_llm(streaming=True)
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def ingest_document(self, file_path: Path) -> int:
        file_hash = self.embedding_service.compute_file_hash(str(file_path))
        cached = self.embedding_service.is_cached(file_hash)
        if cached is not None:
            return len(cached)

        filename, text = load_document(file_path)
        chunks = self.chunker.chunk_text(text, source_path=filename)
        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)
        chunk_ids = self.vector_store.add_chunks(chunk_texts, embeddings, metadatas)
        self.embedding_service.mark_cached(file_hash, chunk_ids)
        return len(chunks)

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[str],
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}] {text}" for i, text in enumerate(retrieved_chunks)
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": DIRECT_RAG_SYSTEM_PROMPT},
        ]
        recent = chat_history[-5:] if len(chat_history) > 5 else chat_history
        for turn in recent:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({
            "role": "user",
            "content": f"## Context\n\n{context}\n\n## Question\n\n{question}",
        })
        return messages

    def _search_with_mmr(
        self, query_embedding: List[float], top_k: int
    ) -> List[SearchResult]:
        fetch_k = max(MMR_FETCH_K, top_k)
        candidates = self.vector_store.search(query_embedding, top_k=fetch_k)
        if len(candidates) <= top_k:
            return candidates
        candidate_texts = [r.text for r in candidates]
        candidate_embeddings = self.embedding_service.embed_texts(candidate_texts)
        return mmr_rerank(
            query_embedding, candidates, candidate_embeddings,
            top_k=top_k, lambda_param=MMR_LAMBDA,
        )

    def query_stream(
        self, question: str, chat_history: List[Dict[str, str]],
        top_k: int = TOP_K,
    ) -> Generator[str, None, None]:
        query_embedding = self.embedding_service.embed_query(question)
        results = self._search_with_mmr(query_embedding, top_k)
        retrieved_texts = [r.text for r in results]
        messages = self.build_prompt(question, retrieved_texts, chat_history)
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def get_retrieved_contexts(
        self, question: str, top_k: int = TOP_K
    ) -> List[dict]:
        query_embedding = self.embedding_service.embed_query(question)
        results = self._search_with_mmr(query_embedding, top_k)
        return [
            {
                "text": r.text[:500],
                "source": r.metadata.get("source", "unknown"),
                "score": round(1.0 - r.score, 4),
            }
            for r in results
        ]

    def list_ingested_sources(self) -> List[str]:
        return self.vector_store.list_sources()

    def delete_document(self, source_path: str) -> int:
        return self.vector_store.delete_by_source(source_path)
