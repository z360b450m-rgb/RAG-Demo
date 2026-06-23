"""Qdrant Cloud-backed vector store.

Preserves the previous VectorStore interface so callers (pipeline, tools)
don't change. Cosine similarity → distance semantics (score = 1 - sim)
to match the existing MMR / display code.
"""

import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException


def _retry(max_attempts: int = 4, base_delay: float = 0.5):
    """Retry Qdrant calls on transient SSL / network drops."""

    def deco(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            last_exc = None
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except (ResponseHandlingException, OSError) as e:
                    last_exc = e
                    if i < max_attempts - 1:
                        time.sleep(base_delay * (2 ** i))
                        continue
                    raise
            raise last_exc

        return wrap

    return deco

from app_config import (
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # distance (lower = more relevant), in [0, 2] for cosine


class VectorStore:
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        vector_size: int = EMBEDDING_DIM,
    ):
        # Long timeout + HTTP (avoid gRPC quirks through Chinese networks)
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
            prefer_grpc=False,
        )
        self.collection_name = collection_name
        self._ensure_collection(vector_size)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    @_retry()
    def _ensure_collection(self, vector_size: int):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        # Index the `source` payload field so filter-by-source is fast
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    @_retry()
    def add_chunks(
        self,
        chunk_texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if chunk_ids is None:
            chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]

        points = [
            qm.PointStruct(
                id=cid,
                vector=vec,
                payload={**meta, "text": text},
            )
            for cid, text, vec, meta in zip(
                chunk_ids, chunk_texts, embeddings, metadata_list
            )
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return chunk_ids

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @_retry()
    def search(
        self, query_embedding: List[float], top_k: int = TOP_K
    ) -> List[SearchResult]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        out: List[SearchResult] = []
        for p in result.points:
            payload = p.payload or {}
            text = payload.pop("text", "")
            # Qdrant returns similarity (higher=better) for cosine.
            # Convert to distance so existing code (MMR, display) works.
            distance = 1.0 - float(p.score)
            out.append(
                SearchResult(
                    chunk_id=str(p.id),
                    text=text,
                    metadata=payload,
                    score=distance,
                )
            )
        return out

    @_retry()
    def get_all_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Return every chunk for a given source, with text + metadata."""
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="source",
                    match=qm.MatchValue(value=source_name),
                )
            ]
        )
        results: List[Dict[str, Any]] = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("text", "")
                results.append({"text": text, "metadata": payload})
            if next_offset is None:
                break
        return results

    @_retry()
    def list_sources(self) -> List[str]:
        sources: set = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )
            for p in points:
                src = (p.payload or {}).get("source")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
        return sorted(sources)

    @_retry()
    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "count": info.points_count,
            "vector_size": info.config.params.vectors.size,
        }

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------
    @_retry()
    def delete_by_source(self, source_path: str) -> int:
        # Count first (Qdrant delete doesn't return count)
        existing = self.get_all_by_source(source_path)
        n = len(existing)
        if n == 0:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source",
                            match=qm.MatchValue(value=source_path),
                        )
                    ]
                )
            ),
        )
        return n
