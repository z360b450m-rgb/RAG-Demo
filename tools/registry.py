"""Tool registry — LangChain-standard @tool decorator.

Two RAG tools:
  1. query_local_knowledge_base  — vector similarity search (semantic)
  2. read_entire_document        — pull all chunks for a file by name
"""

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

    # ------------------------------------------------------------------
    # Tool 1: semantic similarity search
    # ------------------------------------------------------------------
    @tool
    def query_local_knowledge_base(query: str) -> str:
        """Search the local document knowledge base using semantic similarity.
        Use this when the user asks a topical question about uploaded
        documents — e.g. 'what does the report say about revenue?'."""
        query_embedding = embedding_service.embed_query(query)
        results = vector_store.search(query_embedding, top_k=TOP_K)
        if not results:
            return "[No relevant documents found in the knowledge base.]"
        parts = []
        for r in results:
            source = r.metadata.get("source", "unknown")
            chunk_idx = r.metadata.get("chunk_index", "?")
            total = r.metadata.get("total_chunks", "?")
            parts.append(
                f"[Source: {source}  chunk {chunk_idx}/{total}]\n{r.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tool 2: fetch EVERY chunk for a given file (bypasses vector search)
    # ------------------------------------------------------------------
    @tool
    def read_entire_document(source_name: str) -> str:
        """Pull the COMPLETE text of a single document by filename.
        Use this when the user asks for a full summary, wants to read an
        entire document, or when vector search returns fragmented results.
        `source_name` must match one of the indexed filenames exactly
        (e.g. 'report.pdf', 'notes.txt')."""
        rows = vector_store.get_all_by_source(source_name)
        if not rows:
            return (
                f"[File '{source_name}' not found. "
                f"Available sources: {', '.join(vector_store.list_sources())}]"
            )

        # Reassemble chunks in their original order
        indexed = [
            (r["metadata"].get("chunk_index", 0), r["text"]) for r in rows
        ]
        indexed.sort(key=lambda x: x[0])
        full_text = "\n".join(chunk for _, chunk in indexed)

        return (
            f"[Full text of '{source_name}' "
            f"({len(indexed)} chunks)]\n\n{full_text}"
        )

    tools: List = [query_local_knowledge_base, read_entire_document]

    # ------------------------------------------------------------------
    # Optional: web search (requires TAVILY_API_KEY)
    # ------------------------------------------------------------------
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
