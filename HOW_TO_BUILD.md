# 从零搭建 RAG-Demo · 完整流程清单

把这套 **RAG-Demo（Streamlit UI + FastAPI 后端 + LangChain Agent + DeepSeek + SiliconFlow + Qdrant Cloud）** 从空白机器一步一步搭起来要做什么，全在这。按顺序读。

---

## 〇、项目最终长什么样

- **UI**：Streamlit 单页应用，支持文档上传、对话、模式切换（Agent / Direct RAG）
- **后端**：可选 FastAPI 服务暴露 REST + SSE，前后端拆分时用
- **AI 层**：DeepSeek LLM + SiliconFlow embedding，纯云调用，本地零模型
- **存储**：Qdrant Cloud 向量库，无需自建

部署形态有两种：

| 形态 | 启动命令 | 适用场景 |
|---|---|---|
| **单机 Demo** | `streamlit run app.py` | 个人用 / 演示 / 开发调试 |
| **前后端分离** | `uvicorn server:app` + `streamlit run app.py`（前端走 HTTP） | 多用户 / 部署上云 |

---

## 一、前置软件准备（一次性）

| 软件 | 版本 | 干啥用 | 装法 |
|---|---|---|---|
| **Git** | 任意新版 | 版本控制 | https://git-scm.com/ |
| **Python** | ≥ 3.11 | 后端 + Agent | https://www.python.org/，装时勾 "Add to PATH" |
| **VS Code** | 任意 | 编辑器 | https://code.visualstudio.com/ |
| **Chrome 或 Edge** | 任意 | 测 Streamlit UI | 系统自带 |

### 推荐 VS Code 插件

| 插件 | 作用 |
|---|---|
| Python | 基础 Python 支持 |
| Pylance | 类型检查 |
| Ruff 或 Black | 代码格式化 |
| Even Better TOML | `.env` 高亮 |

### 命令行环境

Windows 推荐 **Git Bash**（Git 自带）或 **PowerShell 7+**。不要用 cmd，中文 emoji 会乱码。

---

## 二、申请外部服务（一次性，3 个账号）

### 2.1 DeepSeek API Key（**必需**，主 LLM）

1. 打开 https://platform.deepseek.com/
2. 注册账号（手机号 / 邮箱）
3. 充值（最少 1 元就够测试半天）
4. 控制台 → API Keys → 创建 → 复制 `sk-xxxxx...`
5. **存好这个 key**

**成本**：`deepseek-chat` 输入 ¥0.5 / 1M token，输出 ¥1.5 / 1M token，日常对话一天几分钱。

---

### 2.2 SiliconFlow API Key（**必需**，Embedding）

1. 打开 https://cloud.siliconflow.cn/
2. 注册账号
3. 账户管理 → API 密钥 → 新建 → 复制 `sk-xxxxx...`
4. 模型市场搜 `BAAI/bge-large-zh-v1.5` 确认可用

**成本**：BGE 模型**免费**调用，有 RPM 限制但足够小规模用。

---

### 2.3 Qdrant Cloud Cluster（**必需**，向量库）

1. 打开 https://cloud.qdrant.io/
2. 注册账号
3. **重要：选 Region**
   - 国内访问选 `ap-southeast-1` (Singapore) 或 `ap-northeast-1` (Tokyo)
   - **千万别选南美 / 非洲**，跨大洲 SSL 经常断
4. 创建 Free Tier 集群（1GB 免费）
5. 创建时弹窗给你的 **API Key 只显示一次**，立刻保存
6. Dashboard 复制 **Cluster URL**，形如 `https://xxx.region.aws.cloud.qdrant.io`

**成本**：免费 1GB（约 100 万 chunk），足够个人用。超出按使用量计费。

---

### 2.4 LangSmith（**可选**，调试追踪）

1. https://smith.langchain.com 注册
2. Settings → API Keys → 新建 → `lsv2_pt_xxxxx`

不配也能跑，配了能在网页看每次对话的全链路 trace。

---

## 三、初始化项目

```bash
mkdir -p E:/01_Dev_Projects/Vibe_Coding && cd E:/01_Dev_Projects/Vibe_Coding
git clone https://github.com/z360b450m-rgb/RAG-Demo.git rag-agent
cd rag-agent
```

或者从零搭：

```bash
mkdir rag-agent && cd rag-agent
```

后面假定你在 `rag-agent/` 目录里。

---

## 四、安装 Python 依赖

### 4.1 建虚拟环境（推荐）

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# PowerShell:
.venv\Scripts\Activate.ps1
```

### 4.2 装依赖

```bash
pip install -r requirements.txt
```

如果是从零搭，新建 `requirements.txt`：

```txt
# LLM + Agent
openai>=1.0.0,<2.0.0
langchain>=1.0.0,<2.0.0
langchain-openai>=0.2.0,<1.0.0
langchain-core>=0.3.0,<1.0.0
langchain-text-splitters>=0.3.0,<1.0.0

# Vector DB
qdrant-client>=1.10.0,<2.0.0

# Backend
fastapi>=0.110.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
python-multipart>=0.0.9,<1.0.0
sse-starlette>=2.0.0,<3.0.0
httpx>=0.27.0,<1.0.0

# Frontend
streamlit>=1.31.0,<2.0.0

# Document ingestion
pypdf>=4.0.0,<5.0.0

# Utilities
python-dotenv>=1.0.0,<2.0.0
langsmith>=0.1.0,<1.0.0
```

然后 `pip install -r requirements.txt`。

---

## 五、配置 `.env`

复制模板：

```bash
cp .env.example .env
```

编辑 `.env` 填入四个值：

```bash
# DeepSeek
DEEPSEEK_API_KEY=sk-你的deepseek-key

# SiliconFlow
SILICONFLOW_API_KEY=sk-你的siliconflow-key

# Qdrant Cloud
QDRANT_URL=https://你的集群.region.aws.cloud.qdrant.io
QDRANT_API_KEY=你的qdrant-key

# 可选：LangSmith
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_pt_xxx
# LANGCHAIN_PROJECT=RAG_Agent
```

**重要**：`.env` **绝对不能 commit**，确认 `.gitignore` 里有它。

---

## 六、写代码（如果是从零搭）

按这个顺序建文件，每一步都跑一次最小验证。

### 6.1 `app_config/__init__.py`

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    missing = [k for k, v in {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "SILICONFLOW_API_KEY": SILICONFLOW_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
```

**验证**：
```bash
python -c "from app_config import validate_config; validate_config(); print('OK')"
```

---

### 6.2 `core/llm.py`

完整代码看仓库。关键 3 段：

```python
from langchain_openai import ChatOpenAI
from openai import OpenAI

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
        resp = self.client.embeddings.create(model=self.model, input=[q])
        return resp.data[0].embedding

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
```

**验证**：
```bash
python -c "from core.llm import EmbeddingService; print(len(EmbeddingService().embed_query('你好')))"
# 应该输出 1024
```

---

### 6.3 `database/vector_store.py`

完整代码看仓库。关键点：

- 用 `qdrant-client` 的 `QdrantClient(url=..., api_key=..., timeout=60, prefer_grpc=False)`
- 启动时 `_ensure_collection(vector_size=1024, distance=COSINE)`
- 每个方法包 `@_retry()` 装饰器吸收 SSL 抖动
- `search()` 返回 distance = 1 - similarity（让上层 MMR 代码不变）
- `get_all_by_source()` 用 `scroll()` 分页拉所有 chunks（给 read_entire_document 工具用）

**验证**：
```bash
python -c "from database.vector_store import VectorStore; print(VectorStore().get_collection_stats())"
# 应该输出 {'name': 'rag_documents', 'count': 0, 'vector_size': 1024}
```

---

### 6.4 `database/chunker.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

class TextChunker:
    def __init__(self, chunk_size=800, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    def chunk_text(self, text: str, source_path: str = "") -> list[Chunk]:
        splits = self.splitter.split_text(text)
        return [
            Chunk(
                text=s,
                metadata={"source": source_path, "chunk_index": i, "total_chunks": len(splits)},
            )
            for i, s in enumerate(splits) if s.strip()
        ]
```

---

### 6.5 `database/document_loader.py`

```python
from pathlib import Path

def load_document(file_path) -> tuple[str, str]:
    fp = Path(file_path)
    suffix = fp.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(fp))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
    elif suffix in (".txt", ".md"):
        text = fp.read_bytes().decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported: {suffix}")
    if not text.strip():
        raise ValueError("Empty document")
    return fp.name, text.strip()
```

---

### 6.6 `core/prompt.py`

```python
AGENT_SYSTEM_PROMPT = """You are a precise AI assistant with access to a local knowledge base.

You have two tools:
- query_local_knowledge_base(query) — semantic similarity search
- read_entire_document(source_name) — fetch full text by filename

Rules:
1. Casual chat: answer directly, no tools.
2. Topical questions about documents: use query_local_knowledge_base.
3. Full-document requests: use read_entire_document.
4. If tools return nothing, say so honestly — never invent facts.
5. Cite source filenames when provided."""

DIRECT_RAG_SYSTEM_PROMPT = """Answer based ONLY on the provided context. If insufficient, say so."""
```

---

### 6.7 `tools/registry.py`

```python
from langchain_core.tools import tool

def build_tools(embedding_service, vector_store, enable_web=False):
    @tool
    def query_local_knowledge_base(query: str) -> str:
        """Search the knowledge base by semantic similarity."""
        vec = embedding_service.embed_query(query)
        results = vector_store.search(vec, top_k=15)
        if not results:
            return "[No matches]"
        return "\n\n---\n\n".join(
            f"[Source: {r.metadata.get('source')}]\n{r.text}" for r in results
        )

    @tool
    def read_entire_document(source_name: str) -> str:
        """Fetch the complete text of a single file by name."""
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

### 6.8 `core/agent.py`

**关键：用 `create_agent` 不用自己装 LangGraph**。

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class RAGAgent:
    def __init__(self, embedding_service=None, vector_store=None):
        from core.llm import EmbeddingService
        from database.vector_store import VectorStore
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

    def query_stream(self, user_input: str):
        messages = [*self.chat_history, HumanMessage(content=user_input)]
        printed = ""
        for chunk, _meta in self._graph.stream({"messages": messages}, stream_mode="messages"):
            if isinstance(chunk, AIMessage) and chunk.content:
                new = chunk.content[len(printed):]
                if new:
                    yield new
                printed = chunk.content
        # 关键：流结束后用 invoke 拿规范化的完整历史
        final = self._graph.invoke({"messages": messages})
        self.chat_history = final["messages"]

    def clear_memory(self):
        self.chat_history = []
```

**验证**：
```bash
python -c "from core.agent import RAGAgent; print(RAGAgent().run('你好'))"
```

---

### 6.9 `app.py`（Streamlit UI）

完整代码看仓库，关键结构：

```python
import streamlit as st
from core.agent import RAGAgent
from core.pipeline import DirectRAGPipeline

# 初始化 session_state（agent / pipeline / chat_history）
# 渲染 sidebar（模式切换 / 文档上传 / 文档列表 / 清空历史）
# 渲染 chat（遍历 chat_history 显示）
# 处理输入（调 agent.query_stream 或 pipeline.query_stream，用 st.write_stream）
```

---

### 6.10 `server.py`（FastAPI 后端）

完整代码看仓库，关键结构：

```python
from fastapi import FastAPI, UploadFile, File
from sse_starlette.sse import EventSourceResponse

app = FastAPI()
# 单例化 EmbeddingService / VectorStore / DirectRAGPipeline
# 每个 session_id 一个 RAGAgent
# 端点：/health /sources /sources/{name} /sessions/{sid}/clear /ingest /chat /chat/stream
```

---

## 七、启动顺序

### 7.1 单机 Demo（最快）

```bash
streamlit run app.py
```

浏览器自动打开 http://localhost:8501

### 7.2 前后端分离

```bash
# Terminal 1：后端
uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2：前端（环境变量指向后端）
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

后端 Swagger 文档：http://localhost:8000/docs

### 7.3 CLI 单次调用

```bash
python main.py --query "硅基流动是什么？"
python main.py --ingest mydoc.pdf
python main.py --list
```

---

## 八、验证清单

跑通后挨个测：

1. **配置正确**：`python -c "from app_config import validate_config; validate_config()"` 不报错
2. **Embedding 通**：`python -c "from core.llm import EmbeddingService; print(len(EmbeddingService().embed_query('test')))"` 输出 1024
3. **Qdrant 通**：`python -c "from database.vector_store import VectorStore; print(VectorStore().get_collection_stats())"` 输出统计
4. **Agent 通**：`python -c "from core.agent import RAGAgent; print(RAGAgent().run('你好'))"` 输出问候
5. **Streamlit 起得来**：`streamlit run app.py` 浏览器能打开
6. **文档上传**：在 UI 上传一个 PDF，看到 "X chunks indexed"
7. **问答能跑**：问与上传文档相关的问题，看 LLM 引用文档来源
8. **多轮对话**：连续 5+ 轮包含工具调用的问答，不报 400 错误

---

## 九、常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `ImportError: cannot import name 'X' from 'config'` | 旧 `__pycache__/config.cpython-XXX.pyc` 残留 | `find . -name __pycache__ -exec rm -rf {} +` |
| `SSL UNEXPECTED_EOF_WHILE_READING` 连 Qdrant | 集群在跨大洲区域 | 改建到亚太区，或换自建 |
| `Messages with role 'tool' must be a response to...` | `chat_history` 缺中间 tool 消息 | 用 `invoke()` 拿规范化历史，别手动拼 |
| `UnicodeEncodeError: 'gbk' codec can't encode '\u...'` | Windows 终端 GBK 不认 emoji | `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')` |
| Streamlit 启动卡在 "loading model" | 第一版还在用 sentence-transformers | 检查 `core/llm.py` 是不是已经切到 SiliconFlow |
| LangChain 报 `AgentExecutor not found` | 你装的是 0.x 版本 | `pip install -U "langchain>=1.0"` |
| Qdrant 集合维度不对 | 改过 EMBEDDING_MODEL | 删集合重建：`vs.client.delete_collection("rag_documents")` |
| 提示 `Missing env vars` | `.env` 没填全 | 检查 4 个必填项 |
| LLM 答非所问 | system prompt 没说清楚什么时候用工具 | 调 `core/prompt.py` 的 AGENT_SYSTEM_PROMPT |

---

## 十、Git 工作流

### 10.1 `.gitignore` 必须包含

```
.env
.venv/
venv/
__pycache__/
*.pyc
.streamlit/
data/
*.egg-info/
```

### 10.2 首次推 GitHub

```bash
git init
git add .
git commit -m "init"
gh repo create your-name/RAG-Demo --public --source=. --push
```

### 10.3 永远不要 commit `.env`

```bash
# 推之前扫一遍
grep -rE "sk-[a-zA-Z0-9]{30,}" --include="*.py" --include="*.md" .
# 应该只在 .env 出现（被 ignore 了），其他都是占位符
```

如果不小心 commit 了 key：立即 revoke 那个 key，再用 `git filter-repo` 清历史。

---

## 十一、每天用的启动顺序

```bash
cd E:/01_Dev_Projects/Vibe_Coding/rag-agent
source .venv/Scripts/activate   # 或 .venv\Scripts\Activate.ps1

# 单机模式
streamlit run app.py

# 完事 Ctrl+C
```

数据在 Qdrant Cloud，不会丢。重启 Streamlit 后历史会清（在内存里），上传过的文档还在云端。

---

## 十二、调试技巧

| 场景 | 工具 |
|---|---|
| 看 Agent 内部决策（调没调工具 / 调了哪个） | `core/agent.py` 里把 `create_agent(..., debug=True)` |
| 看每次 LLM 调用 token 数 | 开 LangSmith，网页看 |
| 看 SiliconFlow 返回 | 在 `EmbeddingService` 里 print resp.usage |
| 看 Qdrant 写入 | LangSmith 看不到，得自己在 `add_chunks` print |
| Streamlit 报错 | 浏览器 F12 + Streamlit 终端 traceback |
| FastAPI 报错 | `uvicorn ... --log-level debug` |

---

## 十三、把项目复制到别的电脑

1. 装 Git / Python 3.11+
2. `git clone https://github.com/你的-org/RAG-Demo`
3. `python -m venv .venv && source .venv/Scripts/activate`
4. `pip install -r requirements.txt`
5. 复制 `.env`（或 `.env.example` → `.env` 填新的 key）
6. `streamlit run app.py`

完事。云上 Qdrant 的数据全部同步过来（同一个 cluster）。

---

## 十四、扩展看哪里

`TECH_OVERVIEW.md` 第七章「扩展实施指南」—— 10 个常见扩展（LangSmith / Web 搜索 / 多用户 / Docker / Ragas / OCR / 多模态 / 自建 Qdrant ...），每个都标了改哪些文件、工程量、关键代码点。

---

## 十五、备忘：项目核心命令

```bash
# 装依赖
pip install -r requirements.txt

# 单机起 UI
streamlit run app.py

# 起后端
uvicorn server:app --host 0.0.0.0 --port 8000

# CLI 用法
python main.py --query "..."
python main.py --ingest path/to/file.pdf
python main.py --list

# 清缓存
find . -name __pycache__ -type d -exec rm -rf {} +

# Qdrant 集合重置（开发期）
python -c "from database.vector_store import VectorStore; vs = VectorStore(); vs.client.delete_collection('rag_documents')"
```
