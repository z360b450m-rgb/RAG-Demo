"""LLM and Embedding clients — both call remote APIs.

- LLM: DeepSeek (OpenAI-compatible) via LangChain ChatOpenAI
- Embeddings: SiliconFlow (OpenAI-compatible) via openai SDK
"""

from typing import List

from langchain_openai import ChatOpenAI
from openai import OpenAI

from app_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)

# ---------------------------------------------------------------------------
# LangSmith tracing (env-driven; affects all LangChain calls automatically)
# ---------------------------------------------------------------------------
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


# ---------------------------------------------------------------------------
# LLM factory
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
# Embedding Service — calls SiliconFlow over HTTP
# ---------------------------------------------------------------------------
class EmbeddingService:
    """Thin client over SiliconFlow's OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        api_key: str = SILICONFLOW_API_KEY,
        base_url: str = SILICONFLOW_BASE_URL,
        model: str = EMBEDDING_MODEL,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of texts. Batches to keep request size under control."""
        if not texts:
            return []
        out: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # OpenAI SDK returns data sorted by request index
            out.extend([d.embedding for d in resp.data])
        return out

    def embed_query(self, query: str) -> List[float]:
        # BGE models are asymmetric — query prefix is recommended.
        # SiliconFlow handles this server-side for BGE; we send raw text.
        resp = self.client.embeddings.create(model=self.model, input=[query])
        return resp.data[0].embedding
