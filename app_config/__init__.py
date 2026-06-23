import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM (DeepSeek) ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- Embeddings (SiliconFlow, OpenAI-compatible) ---
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # BGE-large-zh = 1024

# --- Vector DB (Qdrant Cloud) ---
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# --- RAG parameters ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "15"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "40"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))

# --- Backend / Frontend wiring ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- LangSmith (optional tracing) ---
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "RAG_Agent")


def validate_config():
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not SILICONFLOW_API_KEY:
        missing.append("SILICONFLOW_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if missing:
        raise ValueError(
            f"Missing env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
