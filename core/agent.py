"""RAG Agent — powered by LangChain create_agent (v1.3+).

LangChain's CompiledStateGraph handles the full Thought → Action →
Observation loop internally, including tool_calls / tool_call_id pairing.
No more manual dict splicing — we work with HumanMessage, AIMessage,
ToolMessage objects exclusively.
"""

from typing import Generator, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from core.llm import EmbeddingService, build_llm
from core.prompt import AGENT_SYSTEM_PROMPT
from database.vector_store import VectorStore
from tools.registry import build_tools


class RAGAgent:
    """Agent backed by LangChain's create_agent (CompiledStateGraph)."""

    def __init__(
        self,
        enable_web_search: bool = False,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStore = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

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
    # Streaming query — streams tokens, then syncs full history
    # ------------------------------------------------------------------
    def query_stream(self, user_input: str) -> Generator[str, None, None]:
        input_messages = [*self.chat_history, HumanMessage(content=user_input)]
        printed = ""

        # Phase 1: stream tokens for real-time UX
        for chunk, _metadata in self._graph.stream(
            {"messages": input_messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                new = chunk.content[len(printed):]
                if new:
                    yield new
                printed = chunk.content

        # Phase 2: invoke to get the COMPLETE, correctly-ordered message
        # history.  stream(mode="messages") only gives us token-level
        # AIMessageChunks — it doesn't expose the full AIMessage(tool_calls)
        # or ToolMessage objects.  invoke() returns every message in order:
        #   HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage
        final_result = self._graph.invoke({"messages": input_messages})
        self.chat_history = final_result["messages"]

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
