"""RAG Agent — powered by LangChain create_agent (v1.3+).

LangChain's CompiledStateGraph handles the full Thought → Action →
Observation loop internally, including tool_calls / tool_call_id pairing.
No more manual dict splicing — we work with HumanMessage, AIMessage,
ToolMessage objects exclusively.
"""

from typing import Generator, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.llm import EmbeddingService, build_llm
from core.prompt import AGENT_SYSTEM_PROMPT
from database.vector_store import VectorStore
from tools.registry import build_tools


class RAGAgent:
    """Agent backed by LangChain's create_agent (CompiledStateGraph)."""

    def __init__(self, enable_web_search: bool = False):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        self.tools = build_tools(
            self.embedding_service, self.vector_store, enable_web=enable_web_search
        )

        self.llm = build_llm(streaming=True)

        # create_agent returns a CompiledStateGraph — it runs the
        # tool-calling loop and guarantees valid message ordering.
        self._graph = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        # Conversation history as LangChain BaseMessage objects
        self.chat_history: List[BaseMessage] = []

    # ------------------------------------------------------------------
    # Non-streaming run
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        messages = [*self.chat_history, HumanMessage(content=user_input)]
        result = self._graph.invoke({"messages": messages})
        all_msgs: List[BaseMessage] = result["messages"]
        final = all_msgs[-1]
        output = self._extract_content(final)
        self.chat_history = all_msgs
        return output

    # ------------------------------------------------------------------
    # Streaming query — yields text tokens for Streamlit write_stream
    # ------------------------------------------------------------------
    def query_stream(self, user_input: str) -> Generator[str, None, None]:
        messages = [*self.chat_history, HumanMessage(content=user_input)]
        collected = ""

        for chunk, _metadata in self._graph.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                new = chunk.content[len(collected):]
                if new:
                    yield new
                collected = chunk.content

            # Reached terminal AIMessage (no tool_calls) → stop
            if isinstance(chunk, AIMessage) and not chunk.tool_calls:
                break

        # Build final history from the full response
        final_msg = AIMessage(content=collected)
        self.chat_history = [
            *self.chat_history,
            HumanMessage(content=user_input),
            final_msg,
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_content(msg: BaseMessage) -> str:
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part["text"] if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_retrieved_contexts(self, question: str) -> List[dict]:
        from app_config import TOP_K

        query_embedding = self.embedding_service.embed_query(question)
        results = self.vector_store.search(query_embedding, top_k=TOP_K)
        return [
            {
                "text": r.text[:500],
                "source": r.metadata.get("source", "unknown"),
                "score": round(1.0 - r.score, 4),
            }
            for r in results
        ]

    def clear_memory(self):
        self.chat_history = []
