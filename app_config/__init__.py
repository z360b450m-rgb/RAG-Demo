import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
EMBEDDING_CACHE_PATH = str(DATA_DIR / "embedding_cache.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K = 8
MMR_FETCH_K = 20
MMR_LAMBDA = 0.7

# LangSmith
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "RAG_Agent")

DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_config():
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY not set. Create a .env file with: DEEPSEEK_API_KEY=your-key"
        )
