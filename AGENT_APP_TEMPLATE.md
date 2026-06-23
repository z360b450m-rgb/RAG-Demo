# 云原生 RAG Agent · 通用框架蓝本

> **这是什么**：一套搭建「LangChain Agent + 云端 LLM + 云端 Embedding + 云端向量库 + 双形态前端」项目的通用模板。
>
> **怎么用**：把这份文档直接喂给 AI（Claude / GPT / DeepSeek），让它按这个框架搭你自己的项目。
> 替换文档中 `<占位符>` 为你的业务即可。
>
> **来源**：基于 `RAG-Demo` 实际落地的生产骨架提炼。与本目录的 `notes-app` 模板（本地 LangGraph + Chroma + Electron）形成互补，二选一。

---

## 〇、给 AI 助手的 SYSTEM PROMPT

> 复制下面这一整段到 AI 对话开头，它就知道按这个框架干活。

```
你是一位资深 AI 全栈工程师，要按照《云原生 RAG Agent 通用框架蓝本》搭一个新项目。

工作原则：
1. 严格遵守本文档定义的目录结构、命名、依赖版本约束
2. 优先复用框架中给定的代码骨架，不要随意改架构
3. 用 LangChain 1.x 的 create_agent，不要自己装 LangGraph 节点
4. 不要在本地跑任何模型，所有 LLM / Embedding 走云端 OpenAI 兼容 API
5. 向量库默认 Qdrant Cloud，提供切换到自建的最小改造
6. 多轮对话历史用 _graph.invoke() 拿规范化版本，绝不手动拼 messages
7. 任何 API key 必须走 .env，绝不硬编码、绝不写进 .env.example
8. 每完成一个模块跑一次最小验证（python -c "..." 一行能验证完）
9. 遇到选择题主动用 AskUserQuestion 让用户决策，不要替用户选

我的项目主题：<在这里填你的业务>，例如「企业内部 IT 知识库问答 / 法律条文检索 / 学术论文助理」。
我的数据形态：<描述你的核心数据对象>，例如「PDF 文档 / Markdown 笔记 / 网页爬取的纯文本」。
特殊要求：<列出非默认的需求>，例如「需要多用户隔离 / 需要 OCR 扫描件 / 需要 web 搜索补充」。

开始之前，请先确认上面三项信息，然后按照框架第一章开始搭建。
```

---

## 一、项目定位与适用场景

### 这个框架适合做的项目

| 场景 | 例子 |
|---|---|
| **私有知识库问答** | 企业 IT 帮助台、产品手册、政策法规 |
| **文档密集型工具** | 论文阅读助手、合同审阅、报告归档检索 |
| **领域专家助手** | 法律 / 医学 / 金融 文档问答 |
| **快速 PoC** | 客户演示用的最小可用 Demo |
| **轻量 SaaS MVP** | 多用户 + 文档上传 + 对话的最小闭环 |

### 不适合的场景

- 离线 / 内网 / 强合规（数据不能出境）→ 用 `notes-app` 模板（本地模型 + Chroma）
- 桌面端发布（不联网也能跑）→ 用 `notes-app` 模板（Electron 壳）
- 重交易 / 强事务 → 传统 OLTP 架构
- 大规模并发 → 这套是 Streamlit / FastAPI 起步，要扛量得换 Gunicorn + Redis

### 与 notes-app 模板的对照

| 维度 | notes-app（本地版） | RAG-Demo（云原生版） |
|---|---|---|
| LLM | DeepSeek API（云）+ 本地 fallback | DeepSeek API（云） |
| Embedding | 本地 sentence-transformers ~100MB 模型 | SiliconFlow 远程 |
| Reranker | 本地 CrossEncoder 模型 | 不用（默认）/ 可选云端 |
| 向量库 | Chroma 本地文件 | Qdrant Cloud |
| 关键词检索 | BM25 本地 | 不用（默认）/ 可选 Qdrant BM42 |
| Agent 编排 | LangGraph 手装节点 | LangChain create_agent 内置状态机 |
| 前端 | Vue 3 + Electron 桌面壳 | Streamlit Web UI |
| 部署 | 用户本地双进程 | 云上 FastAPI + Streamlit |
| 启动成本 | 装 Node + Python + 下模型 | 装 Python + 3 个云账号 |
| 适合 | 离线 / 数据敏感 / 桌面体验 | 快速演示 / 云上 SaaS / 多用户 |

### 核心特征

```
┌──────────────────────┐         ┌──────────────────────────────┐
│  前端（任选一种）     │  HTTP   │  Agent 后端                    │
│ • Streamlit Web      │ ─────► │ • FastAPI                      │
│ • 你自己的 React/Vue │ ◄──SSE  │ • LangChain create_agent       │
│ • CLI 工具           │         │ • Tool-calling 状态机自动管理  │
└──────────────────────┘         └──────────────┬───────────────┘
                                                │
                       ┌────────────────────────┼─────────────────────────┐
                       ▼                        ▼                         ▼
                ┌─────────────┐         ┌──────────────┐         ┌─────────────┐
                │ LLM 云 API  │         │ Embedding 云 │         │ Vector DB 云 │
                │  DeepSeek / │         │ SiliconFlow /│         │   Qdrant /   │
                │  Qwen /     │         │   阿里百炼  /│         │   自建 /    │
                │  GPT...     │         │   OpenAI    │         │   Pinecone  │
                └─────────────┘         └──────────────┘         └─────────────┘
```

---

## 二、技术栈（默认选型）

### 后端

| 类别 | 技术 | 替代项 |
|---|---|---|
| 语言 | Python 3.11+ | - |
| Web 框架 | FastAPI + Uvicorn | Flask（轻量） / Hono |
| Agent 编排 | **LangChain 1.3+ `create_agent`** | LangGraph 自装 / 原生 OpenAI tools loop |
| LLM | DeepSeek（OpenAI 兼容） | Qwen / GPT / Claude / 月之暗面 |
| Embedding | SiliconFlow `bge-large-zh-v1.5` | 阿里百炼 / OpenAI / Cohere |
| 向量库 | Qdrant Cloud | 自建 Qdrant / Pinecone / Weaviate / 阿里 DashVector |
| 文档切分 | LangChain `RecursiveCharacterTextSplitter` | 自写正则 / unstructured |
| 文档解析 | pypdf | unstructured / pdfplumber / PyMuPDF |
| 流式 | sse-starlette | WebSocket / chunked transfer |
| 校验 | Pydantic 2 | - |

### 前端（任选一种）

| 形态 | 技术 | 何时用 |
|---|---|---|
| **Streamlit**（默认） | streamlit 1.31+ | 内部工具 / Demo / 个人用 |
| **静态 + 自己写** | 任意 SPA + fetch | 要对外发布 / 多用户 |
| **CLI** | argparse + httpx | 嵌入到管道 / 自动化 |
| **MCP Server** | mcp Python SDK | 接入 Claude Desktop / Cline |

### 工程化

- 单文件 `.env` 管所有 key
- `requirements.txt` 钉版本范围（不是固定版本）
- Git Hook（可选）：pre-commit 扫 `sk-` 防 key 泄露
- LangSmith 一键开启 trace

---

## 三、目录结构（强约束）

```
<project-root>/
├── README.md
├── TECH_OVERVIEW.md            # 技术档案（每个项目都写）
├── HOW_TO_BUILD.md             # 复现指南
│
├── app.py                      # Streamlit UI（或换成 cli.py）
├── server.py                   # FastAPI 后端（前后端分离时启用）
├── main.py                     # CLI 入口
├── requirements.txt
├── .env                        # 密钥 ❗ 加 .gitignore
├── .env.example                # 模板，key 用占位符
├── .gitignore
│
├── app_config/
│   └── __init__.py             # 环境变量集中读取 + 工厂用常量
│
├── core/                       # Agent 核心
│   ├── __init__.py
│   ├── agent.py                # RAGAgent / 业务 Agent 类
│   ├── pipeline.py             # 不带工具决策的强制 RAG 流程（可选）
│   ├── llm.py                  # LLM + Embedding 工厂
│   └── prompt.py               # System prompt 集中管理
│
├── tools/                      # 工具箱（@tool 装饰器）
│   ├── __init__.py
│   └── registry.py             # build_tools() 返回 [tool1, tool2, ...]
│
└── database/                   # 数据层
    ├── __init__.py
    ├── vector_store.py         # Qdrant 客户端 + 重试装饰器
    ├── chunker.py              # 文本切分
    └── document_loader.py      # PDF / TXT / MD 加载
```

**注意**：项目根目录的「业务名」可以随便起（`rag-agent` / `support-bot` / `paper-helper`），但**内部包结构强制按上面**——`app_config` / `core` / `tools` / `database`。别用 `config` 作包名（会和 Python stdlib 的 logging.config 之类产生 `.pyc` 缓存冲突）。

---

## 四、核心架构模式

### 4.1 用 `create_agent` 不要自己装 LangGraph（必备）

LangChain 1.3+ 的 `create_agent` 内部就是一个编译好的 `CompiledStateGraph`，已经处理好：
- tool_calls 状态判断
- tool_call_id 配对
- 多轮 tool 调用循环
- 异常恢复

**反模式**：自己写 `StateGraph().add_node().add_edge()` —— 除非你有特殊节点（如 grade_documents 评分），否则纯 RAG + tool-calling 用不到。

```python
from langchain.agents import create_agent

agent = create_agent(llm, tools=[...], system_prompt="...")
result = agent.invoke({"messages": [HumanMessage(...)]})
# result["messages"] 自动包含完整规范的对话链
```

---

### 4.2 流式 + 历史持久化（必备）

LangChain 的 `stream(stream_mode="messages")` 给的是**实时 token**，不是完整消息。流结束后**必须用 `invoke()` 拿规范化历史**，否则 tool_calls / ToolMessage 配对会丢，下一轮 400。

```python
def query_stream(self, user_input: str):
    messages = [*self.chat_history, HumanMessage(content=user_input)]
    printed = ""
    # 阶段 1：流给前端
    for chunk, _meta in self._graph.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(chunk, AIMessage) and chunk.content:
            new = chunk.content[len(printed):]
            if new:
                yield new
            printed = chunk.content
    # 阶段 2：拿规范化历史
    final = self._graph.invoke({"messages": messages})
    self.chat_history = final["messages"]
```

**这是这个框架最容易踩的坑**，必须照抄。

---

### 4.3 工具用 `@tool` 装饰器声明（必备）

```python
from langchain_core.tools import tool

@tool
def query_local_knowledge_base(query: str) -> str:
    """语义检索本地知识库。
    用于：用户问文档里有什么内容、特定概念定义等。
    """
    vec = embedding.embed_query(query)
    results = vector_store.search(vec, top_k=15)
    return "\n\n---\n\n".join(f"[{r.metadata.get('source')}]\n{r.text}" for r in results)
```

**关键**：
- docstring 就是给 LLM 看的工具说明（自动转 JSON Schema）
- 参数类型注解必须给（LLM 看类型判断怎么调）
- 返回字符串（多模态返回得用 LangChain 特殊类型）
- 在 docstring 里说"什么时候用"，不要说"怎么实现"

---

### 4.4 云端 LLM + Embedding（必备）

**统一用 OpenAI 兼容协议**，所有调云端模型都走 `langchain_openai.ChatOpenAI` 或 `openai.OpenAI`：

```python
# LLM
ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com/v1",
)

# Embedding
OpenAI(api_key=SILICONFLOW_KEY, base_url="https://api.siliconflow.cn/v1").embeddings.create(
    model="BAAI/bge-large-zh-v1.5",
    input=texts,
)
```

切换供应商**只改 base_url + model**，代码不动。

---

### 4.5 向量库重试包装（必备）

云端向量库 SSL 抖动是常态。用装饰器统一兜底：

```python
def _retry(max_attempts=4, base_delay=0.5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except (ResponseHandlingException, OSError):
                    if i < max_attempts - 1:
                        time.sleep(base_delay * (2 ** i))
                        continue
                    raise
        return wrap
    return deco

class VectorStore:
    @_retry()
    def search(self, ...): ...
    @_retry()
    def add_chunks(self, ...): ...
```

---

### 4.6 双形态前端（按需）

同一个后端能被三种前端复用：

```python
# 形态 1：Streamlit 直连（单机 Demo）
agent = RAGAgent()
for token in agent.query_stream(question):
    st.write_stream(...)

# 形态 2：FastAPI + 任意前端
async def chat_stream(req):
    return EventSourceResponse(...)   # SSE

# 形态 3：CLI
result = agent.run(question)
print(result)
```

`core/agent.py` 完全和 UI 解耦。

---

### 4.7 Session 隔离（按需）

```python
# server.py
SESSIONS: dict[str, RAGAgent] = {}

def get_agent(sid: str) -> RAGAgent:
    if sid not in SESSIONS:
        SESSIONS[sid] = RAGAgent(
            embedding_service=SHARED_EMBEDDING,  # 复用云端 client
            vector_store=SHARED_VECTOR,
        )
    return SESSIONS[sid]
```

进程内存里。要持久化跨重启 → 把 `chat_history` 用 `langchain_core.load.dumps/loads` 序列化进 Redis。

---

## 五、API 设计约定

| 端点 | 方法 | 用途 |
|---|---|---|
| `/health` | GET | 健康检查 + 数据统计 |
| `/chat` | POST | 同步问答 |
| `/chat/stream` | POST | SSE 流式问答 |
| `/sessions/{sid}/clear` | POST | 清空 session 历史 |
| `/sources` | GET | 列出已索引文件 |
| `/sources/{name}` | DELETE | 删除某文件的 chunks |
| `/ingest` | POST (multipart) | 上传文档 |
| `/docs` | GET | 自动生成的 OpenAPI 文档 |

**请求体**：
```json
{ "session_id": "...", "message": "...", "mode": "agent" }
```

**SSE 事件流**：
```
data: {"token": "..."}\r\n\r\n
data: {"token": "..."}\r\n\r\n
data: {"done": true, "contexts": [...]}\r\n\r\n
```

前端 SSE 解析**必须同时支持 `\r\n\r\n` 和 `\n\n`**（sse-starlette 用前者）。

---

## 六、代码骨架（直接复制）

### 6.1 `app_config/__init__.py`

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "15"))


def validate_config():
    needed = {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "SILICONFLOW_API_KEY": SILICONFLOW_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }
    missing = [k for k, v in needed.items() if not v]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
```

---

### 6.2 `core/llm.py`

```python
from langchain_openai import ChatOpenAI
from openai import OpenAI
from app_config import *

def build_llm(temperature=0.7, streaming=True):
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        self.model = EMBEDDING_MODEL

    def embed_query(self, q: str) -> list[float]:
        return self.client.embeddings.create(model=self.model, input=[q]).data[0].embedding

    def embed_texts(self, texts: list[str], batch_size=32) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
```

---

### 6.3 `core/agent.py`（最关键的文件）

```python
from typing import Generator
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from core.llm import EmbeddingService, build_llm
from core.prompt import AGENT_SYSTEM_PROMPT
from database.vector_store import VectorStore
from tools.registry import build_tools


class RAGAgent:
    def __init__(self, embedding_service=None, vector_store=None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.tools = build_tools(self.embedding_service, self.vector_store)
        self.llm = build_llm(streaming=True)
        self._graph = create_agent(self.llm, tools=self.tools, system_prompt=AGENT_SYSTEM_PROMPT)
        self.chat_history: list[BaseMessage] = []

    def run(self, user_input: str) -> str:
        messages = [*self.chat_history, HumanMessage(content=user_input)]
        result = self._graph.invoke({"messages": messages})
        self.chat_history = result["messages"]
        return self.chat_history[-1].content

    def query_stream(self, user_input: str) -> Generator[str, None, None]:
        messages = [*self.chat_history, HumanMessage(content=user_input)]
        printed = ""
        for chunk, _meta in self._graph.stream({"messages": messages}, stream_mode="messages"):
            if isinstance(chunk, AIMessage) and chunk.content:
                new = chunk.content[len(printed):]
                if new:
                    yield new
                printed = chunk.content
        # 关键：拿规范化历史
        final = self._graph.invoke({"messages": messages})
        self.chat_history = final["messages"]

    def clear_memory(self):
        self.chat_history = []
```

---

### 6.4 `database/vector_store.py`（精简版）

```python
import time, uuid, functools
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException
from app_config import *


def _retry(max_attempts=4, base_delay=0.5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            for i in range(max_attempts):
                try: return fn(*args, **kwargs)
                except (ResponseHandlingException, OSError):
                    if i < max_attempts - 1:
                        time.sleep(base_delay * (2 ** i))
                        continue
                    raise
        return wrap
    return deco


@dataclass
class SearchResult:
    chunk_id: str; text: str; metadata: dict; score: float


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60, prefer_grpc=False)
        self.collection = QDRANT_COLLECTION
        self._ensure_collection()

    @_retry()
    def _ensure_collection(self):
        if self.collection not in {c.name for c in self.client.get_collections().collections}:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
            )
            self.client.create_payload_index(self.collection, "source", qm.PayloadSchemaType.KEYWORD)

    @_retry()
    def add_chunks(self, texts, embeddings, metas, ids=None):
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        points = [qm.PointStruct(id=i, vector=v, payload={**m, "text": t})
                  for i, t, v, m in zip(ids, texts, embeddings, metas)]
        self.client.upsert(self.collection, points)
        return ids

    @_retry()
    def search(self, qv, top_k=TOP_K):
        r = self.client.query_points(self.collection, query=qv, limit=top_k, with_payload=True)
        return [SearchResult(str(p.id), (p.payload or {}).pop("text", ""), p.payload, 1.0 - float(p.score))
                for p in r.points]

    @_retry()
    def get_all_by_source(self, source):
        flt = qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
        out, off = [], None
        while True:
            pts, off = self.client.scroll(self.collection, scroll_filter=flt, limit=256, offset=off, with_payload=True)
            for p in pts:
                payload = dict(p.payload or {})
                out.append({"text": payload.pop("text", ""), "metadata": payload})
            if off is None: break
        return out

    @_retry()
    def list_sources(self):
        sources, off = set(), None
        while True:
            pts, off = self.client.scroll(self.collection, limit=256, offset=off, with_payload=True)
            for p in pts:
                if src := (p.payload or {}).get("source"): sources.add(src)
            if off is None: break
        return sorted(sources)

    @_retry()
    def delete_by_source(self, source):
        rows = self.get_all_by_source(source)
        if not rows: return 0
        self.client.delete(self.collection, points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
        ))
        return len(rows)
```

---

### 6.5 `tools/registry.py`

```python
from langchain_core.tools import tool
from app_config import TOP_K

def build_tools(embedding_service, vector_store):
    @tool
    def query_local_knowledge_base(query: str) -> str:
        """Search the knowledge base by semantic similarity.
        Use for topical questions about documents."""
        vec = embedding_service.embed_query(query)
        results = vector_store.search(vec, top_k=TOP_K)
        if not results:
            return "[No matches]"
        return "\n\n---\n\n".join(
            f"[Source: {r.metadata.get('source')}]\n{r.text}" for r in results
        )

    @tool
    def read_entire_document(source_name: str) -> str:
        """Fetch the complete text of a single file by filename.
        Use for full-document summaries or 'read chapter X' requests."""
        rows = vector_store.get_all_by_source(source_name)
        if not rows:
            return f"[File '{source_name}' not found]"
        indexed = sorted(
            [(r["metadata"].get("chunk_index", 0), r["text"]) for r in rows],
            key=lambda x: x[0],
        )
        return "\n".join(t for _, t in indexed)

    return [query_local_knowledge_base, read_entire_document]
```

---

### 6.6 `server.py` 最小版

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import asyncio, json, tempfile
from pathlib import Path
from app_config import validate_config
from core.agent import RAGAgent
from core.pipeline import DirectRAGPipeline
from core.llm import EmbeddingService
from database.vector_store import VectorStore

validate_config()

EMB = EmbeddingService()
VEC = VectorStore()
PIPE = DirectRAGPipeline()
PIPE.embedding_service = EMB; PIPE.vector_store = VEC

SESSIONS: dict[str, RAGAgent] = {}
def get_agent(sid): 
    if sid not in SESSIONS:
        SESSIONS[sid] = RAGAgent(embedding_service=EMB, vector_store=VEC)
    return SESSIONS[sid]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatReq(BaseModel):
    session_id: str
    message: str
    mode: str = "agent"


@app.get("/health")
def health(): return {"status": "ok", "stats": VEC.get_collection_stats(), "sessions": len(SESSIONS)}

@app.get("/sources")
def sources(): return {"sources": VEC.list_sources()}

@app.delete("/sources/{name}")
def del_source(name: str): return {"deleted": VEC.delete_by_source(name)}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data); fp = Path(tmp.name)
    try: count = PIPE.ingest_document(fp)
    finally: fp.unlink(missing_ok=True)
    return {"filename": file.filename, "chunks_added": count}

@app.post("/chat")
def chat(req: ChatReq):
    agent = get_agent(req.session_id)
    return {"response": agent.run(req.message)}

@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    async def gen():
        loop = asyncio.get_event_loop()
        agent = get_agent(req.session_id)
        q = asyncio.Queue()
        def producer():
            try:
                for tok in agent.query_stream(req.message):
                    loop.call_soon_threadsafe(q.put_nowait, ("token", tok))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))
        loop.run_in_executor(None, producer)
        while True:
            kind, payload = await q.get()
            if kind == "token": yield {"data": json.dumps({"token": payload})}
            elif kind == "error": yield {"data": json.dumps({"error": payload})}; break
            else: yield {"data": json.dumps({"done": True})}; break
    return EventSourceResponse(gen())
```

---

## 七、踩坑清单（每个项目都会遇到）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `Messages with role 'tool' must be a response to...` | 手动拼 chat_history，丢了中间 tool 消息 | 流结束后 `invoke()` 拿规范化历史 |
| 2 | `[SSL: UNEXPECTED_EOF_WHILE_READING]` 连云端向量库 | 集群跨大洲 | 改区域到亚太，或自建 Qdrant Docker |
| 3 | `ImportError: cannot import name 'X' from 'config'` | 旧 `__pycache__/config.cpython-X.pyc` 残留 | 别用 `config` 作包名（用 `app_config`），清缓存 |
| 4 | `AgentExecutor` 找不到 | 装的是 LangChain 0.x | `pip install -U "langchain>=1.0"`，改用 `create_agent` |
| 5 | Qdrant 集合维度不匹配 | 换 Embedding 模型后没重建 | `client.delete_collection(name)` 重新跑 |
| 6 | Streamlit `st.write_stream` 阻塞 | `query_stream` 中间有同步 IO 拖住 | 把 IO 移到生成器外面 |
| 7 | Windows 终端 emoji 报错 | GBK 不认 | `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')` |
| 8 | LLM 答非所问 | system prompt 没把工具说清楚 | docstring + AGENT_SYSTEM_PROMPT 双管齐下 |
| 9 | 工具被无脑调用 | 闲聊也调 query | system prompt 加 "casual chat: answer directly" |
| 10 | `.env` 不小心 commit | gitignore 漏配 | 立即 revoke key + `git filter-repo` 清史 |
| 11 | SiliconFlow 调用返回 401 | Key 拼错 / 多了空格 | 检查 `.env` 末尾换行 |
| 12 | 前端 CORS 错 | FastAPI 没装中间件 | `app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)` |
| 13 | SSE 一直转圈 | 前端按 `\n\n` 切，后端发 `\r\n\r\n` | 解析器同时支持两种 |
| 14 | LLM 调用挂在 stream | 启动时 streaming=False，调用 stream() | `build_llm(streaming=True)` |
| 15 | 上传大 PDF OOM | 一次性 embed | `embed_texts` 加 batch_size=32 |

---

## 八、可选扩展（按需开关）

| 扩展 | 工程量 | 何时做 |
|---|---|---|
| **LangSmith Tracing** | 5 分钟 | 调试 / 量化 token 成本 |
| **Web 搜索工具**（Tavily） | 1 小时 | Agent 需要实时信息 |
| **自建 Qdrant Docker** | 10 分钟 | 不想付云费 / 出境合规 |
| **多用户 + Redis Session** | 1 天 | 真正多用户场景 |
| **Docker Compose 部署** | 半天 | 上线前提 |
| **Ragas 自动评测** | 1 天 | 想量化 RAG 质量 |
| **MCP Server 包装** | 半天 | 想接入 Claude Desktop |
| **OCR 扫描 PDF** | 1-2 天 | 业务有扫描件 |
| **HyDE / 子问题拆解** | 1-2 天 | 召回不准 |
| **多模态（图文）** | 3-5 天 | 业务含图片 |
| **结构化输出** | 半天 | 前端要 JSON 不要 markdown |

---

## 九、安全清单（推 git 前必查）

```bash
# 1. .gitignore 必须有
grep -E "^\.env$" .gitignore || echo "❌ 漏了 .env"

# 2. 扫真实 key 残留
grep -rE "sk-[a-zA-Z0-9]{30,}" --include="*.py" --include="*.md" .
# 只能在 .env（被 ignore 了）出现

# 3. .env.example 必须脱敏
grep -E "sk-|password|secret" .env.example
# 都该是 your-xxx-here

# 4. git status 看清楚
git status
git diff --cached
```

---

## 十、典型对话流程（喂给 AI 时这样开场）

```
你好，我要用《云原生 RAG Agent 通用框架蓝本》搭一个项目。

业务主题：法律条文检索助手
核心数据形态：上传的 PDF 法律法规、案例判决书
特殊要求：
1. 需要按"法规 / 案例"两类独立检索
2. 用户每次问完要给出引用条文编号
3. 部署到云端，多人用

请按框架第三章给我目录结构（业务名为 law-helper），
然后第四章拆解需要哪些工具（除了默认 query / read，还要不要 cite_article），
最后第六章给出 core/agent.py 和 server.py 的具体代码（带上 system prompt 改造）。
完成后告诉我下一步该装什么依赖。
```

AI 会顺着这个 prompt：
1. 输出适配你业务的目录树
2. 列出工具（含 `cite_article(article_id)` 之类）
3. 给出 system prompt 强调"必须引用编号"
4. 给出 `requirements.txt`
5. 提示你下一步装依赖、申请 key

---

## 十一、复盘 · 这套框架解决了什么

| 痛点 | 解决方式 |
|---|---|
| 自己装 LangGraph 节点繁琐易错 | 用 `create_agent` 内置状态机 |
| tool_calls / tool_call_id 配对手动维护 | `_graph.invoke()` 拿规范化历史 |
| 本地模型 100MB+ 模型下载慢 | 全云端，启动 0 等待 |
| 多平台模型兼容（macOS / Windows / Linux）| 全 HTTP API 调用，无平台差异 |
| 向量库本地数据备份难 | Qdrant Cloud 内置备份 |
| 多用户隔离麻烦 | session_id → RAGAgent 字典即可 |
| 前后端拆分改造大 | core/agent.py 与 UI 解耦，三种前端复用 |
| LangChain 版本碎片化 | 钉 1.x，1.x API 稳定 |
| 密钥泄露 | .env + .gitignore + 推前扫描 |
| 跨大洲网络抖动 | `_retry()` 装饰器 + 选近区域 |

---

## 十二、参考资料（持续更新）

- [LangChain `create_agent` 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain 1.x 迁移指南](https://docs.langchain.com/oss/python/migrate)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)
- [SiliconFlow API 文档](https://docs.siliconflow.cn/)
- [Qdrant Cloud 文档](https://qdrant.tech/documentation/cloud/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [FastAPI + SSE](https://fastapi.tiangolo.com/)
- [Streamlit `st.write_stream`](https://docs.streamlit.io/library/api-reference/write-magic/st.write_stream)
- [LangSmith Tracing](https://docs.smith.langchain.com/)

---

> 本框架来源于真实 MVP 项目 `RAG-Demo`，已经过端到端验证（流式 + 多轮 + 工具调用 + 历史持久化）。
> 复用时记得替换业务实体名、调整 system prompt、按你的数据形态选 Embedding 模型。
