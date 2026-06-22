import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from app_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    LANGCHAIN_API_KEY,
    LANGCHAIN_TRACING_V2,
)
from core.prompt import BGE_QUERY_PREFIX

# ---------------------------------------------------------------------------
# LangSmith tracing
# ---------------------------------------------------------------------------
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RAG_Agent")


# ---------------------------------------------------------------------------
# LLM factory — returns a LangChain ChatOpenAI pointed at DeepSeek
# ---------------------------------------------------------------------------
def build_llm(temperature: float = 0.7, streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Embedding Service (keeps sentence-transformers — no LangChain wrapper needed)
# ---------------------------------------------------------------------------
class EmbeddingService:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
        cache_path: str = EMBEDDING_CACHE_PATH,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, List[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        prefixed = f"{BGE_QUERY_PREFIX}{query}"
        embedding = self.model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )
        return embedding[0].tolist()

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        hasher.update(f"|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
        return hasher.hexdigest()

    def is_cached(self, file_hash: str) -> Optional[List[str]]:
        return self._cache.get(file_hash)

    def mark_cached(self, file_hash: str, chunk_ids: List[str]):
        self._cache[file_hash] = chunk_ids
        self._save_cache()

    def remove_from_cache(self, file_hash: str):
        if file_hash in self._cache:
            del self._cache[file_hash]
            self._save_cache()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
