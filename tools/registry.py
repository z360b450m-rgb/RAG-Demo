"""Tool registry — LangChain-standard @tool decorator, no more manual dict wrangling."""

from typing import List

from langchain_core.tools import tool

from app_config import TOP_K
from core.llm import EmbeddingService
from database.vector_store import VectorStore


def build_tools(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    enable_web: bool = False,
) -> List:
    """Build the list of LangChain tools available to the agent."""

    @tool
    def query_local_knowledge_base(query: str) -> str:
        """Search the local document knowledge base for relevant information.
        Use this whenever the user asks about uploaded documents, project
        details, team information, or technical knowledge stored in the system."""
        query_embedding = embedding_service.embed_query(query)
        results = vector_store.search(query_embedding, top_k=TOP_K)
        if not results:
            return "[No relevant documents found in the knowledge base.]"
        parts = []
        for r in results:
            source = r.metadata.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{r.text}")
        return "\n\n---\n\n".join(parts)

    tools: List = [query_local_knowledge_base]

    if enable_web:

        @tool
        def search_web(query: str) -> str:
            """Search the web for recent or real-time information.
            Use this when the user asks about current events, news, weather,
            or any information beyond your knowledge cutoff."""
            return (
                "[Web search is not configured.] "
                "Set TAVILY_API_KEY in .env to enable internet search."
            )

        tools.append(search_web)

    return tools
