import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app_config import CHROMA_DB_PATH, TOP_K


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_path: str = CHROMA_DB_PATH,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunk_texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if chunk_ids is None:
            chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadata_list,
        )
        return chunk_ids

    def search(
        self, query_embedding: List[float], top_k: int = TOP_K
    ) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=results["distances"][0][i],
                ))
        return search_results

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
        }

    def delete_by_source(self, source_path: str) -> int:
        results = self.collection.get(where={"source": source_path}, include=[])
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def list_sources(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        sources = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])
        return sorted(sources)
