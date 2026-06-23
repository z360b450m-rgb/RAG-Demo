"""FastAPI backend for the RAG Agent.

Exposes:
  POST   /chat                 — non-streaming chat
  POST   /chat/stream          — SSE streaming chat
  POST   /ingest               — upload a document
  GET    /sources              — list indexed filenames
  DELETE /sources/{name}       — remove a file's chunks
  POST   /sessions/{sid}/clear — wipe a session's chat history
  GET    /health               — readiness
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app_config import validate_config
from core.agent import RAGAgent
from core.llm import EmbeddingService
from core.pipeline import DirectRAGPipeline
from database.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
validate_config()

# Shared singletons — created once, reused across all requests/sessions
EMBEDDING = EmbeddingService()
VECTOR = VectorStore()
PIPELINE = DirectRAGPipeline()
PIPELINE.embedding_service = EMBEDDING  # reuse singleton
PIPELINE.vector_store = VECTOR

# Per-session agents (in-memory; replace with Redis for multi-instance prod)
SESSIONS: Dict[str, RAGAgent] = {}


def get_agent(session_id: str) -> RAGAgent:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = RAGAgent(
            embedding_service=EMBEDDING, vector_store=VECTOR
        )
    return SESSIONS[session_id]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "agent"  # "agent" | "direct"


class ChatResponse(BaseModel):
    response: str
    contexts: Optional[list] = None


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    stats = VECTOR.get_collection_stats()
    return {"status": "ok", "vector_db": stats, "sessions": len(SESSIONS)}


@app.get("/sources")
def list_sources():
    return {"sources": VECTOR.list_sources()}


@app.delete("/sources/{name}")
def delete_source(name: str):
    n = VECTOR.delete_by_source(name)
    return {"deleted_chunks": n}


@app.post("/sessions/{session_id}/clear")
def clear_session(session_id: str):
    if session_id in SESSIONS:
        SESSIONS[session_id].clear_memory()
    return {"cleared": session_id}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        count = PIPELINE.ingest_document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(filename=file.filename, chunks_added=count)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.mode == "agent":
        agent = get_agent(req.session_id)
        answer = agent.run(req.message)
        return ChatResponse(response=answer, contexts=None)

    # Direct RAG mode — no per-session state
    contexts = PIPELINE.get_retrieved_contexts(req.message)
    tokens = []
    for tok in PIPELINE.query_stream(req.message, chat_history=[]):
        tokens.append(tok)
    return ChatResponse(response="".join(tokens), contexts=contexts)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint. Streams tokens as `data: {"token": "..."}` events,
    then a final `data: {"done": true, "contexts": [...]}` event."""

    async def event_generator():
        loop = asyncio.get_event_loop()

        if req.mode == "agent":
            agent = get_agent(req.session_id)
            # query_stream is a sync generator; run it in a thread so we don't
            # block the event loop, and forward tokens.
            queue: asyncio.Queue = asyncio.Queue()

            def producer():
                try:
                    for token in agent.query_stream(req.message):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

            await loop.run_in_executor(None, lambda: None)  # warm pool
            loop.run_in_executor(None, producer)

            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    yield {"data": json.dumps({"token": payload})}
                elif kind == "error":
                    yield {"data": json.dumps({"error": payload})}
                    break
                else:  # done
                    yield {"data": json.dumps({"done": True})}
                    break

        else:
            # Direct RAG mode
            contexts = PIPELINE.get_retrieved_contexts(req.message)

            def producer(q):
                try:
                    for token in PIPELINE.query_stream(
                        req.message, chat_history=[]
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token))
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, ("done", None))

            queue: asyncio.Queue = asyncio.Queue()
            loop.run_in_executor(None, lambda: producer(queue))

            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    yield {"data": json.dumps({"token": payload})}
                elif kind == "error":
                    yield {"data": json.dumps({"error": payload})}
                    break
                else:
                    yield {
                        "data": json.dumps({"done": True, "contexts": contexts})
                    }
                    break

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Local dev entry point: `python server.py` or `uvicorn server:app --reload`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
