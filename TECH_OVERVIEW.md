# RAG-Demo 完整技术档案

一套**完全云原生**的 RAG Agent 系统：所有模型与向量库都跑在云端，本地零依赖（不下任何模型、不存任何向量）。同时保留了 Streamlit 单机 Demo 模式，可在前后端拆分前直接跑起来用。

---

## 一、项目结构

```
rag-agent/
├── app.py                       # Streamlit UI（直连 / 单机 Demo 模式）
├── server.py                    # FastAPI 后端（前后端分离模式）
├── main.py                      # CLI 入口（python main.py --query "..."）
├── requirements.txt
├── .env.example                 # 环境变量模板
│
├── app_config/
│   └── __init__.py              # 全局配置：API key / 模型 / 路径
│
├── core/                        # Agent 核心
│   ├── agent.py                 # RAGAgent —— LangChain create_agent 包装
│   ├── pipeline.py              # DirectRAGPipeline —— 强制检索模式
│   ├── llm.py                   # LLM + Embedding 工厂
│   └── prompt.py                # 系统提示集中管理
│
├── tools/                       # 工具箱（@tool 装饰器）
│   └── registry.py              # query_local_knowledge_base + read_entire_document
│
└── database/                    # 数据层
    ├── vector_store.py          # Qdrant Cloud 客户端 + 重试
    ├── chunker.py               # RecursiveCharacterTextSplitter
    └── document_loader.py       # PDF / TXT / MD 加载
```

---

## 二、技术栈

### 全栈选型

| 类别 | 技术 | 备注 |
|---|---|---|
| 语言 | **Python 3.11+** | |
| Agent 编排 | **LangChain 1.3+** `create_agent` | 不自己装 LangGraph 节点，框架内置状态机 |
| LLM 客户端 | **langchain-openai** ChatOpenAI | OpenAI 兼容协议接 DeepSeek |
| 主 LLM | **DeepSeek `deepseek-chat`** | `https://api.deepseek.com/v1` |
| Embedding | **SiliconFlow `BAAI/bge-large-zh-v1.5`** | 1024 维，OpenAI 兼容协议 |
| 向量库 | **Qdrant Cloud** | 全托管，无需自建 |
| 文档切分 | **LangChain RecursiveCharacterTextSplitter** | 段落 / 换行 / 句号优先级 |
| 文档解析 | **pypdf** | 文字层 PDF |
| Web 后端 | **FastAPI + Uvicorn** | REST + SSE |
| 流式响应 | **sse-starlette** | Server-Sent Events |
| 前端 | **Streamlit** | 内置 Demo UI |
| 配置 | **python-dotenv** | `.env` 加载 |
| HTTP 客户端 | **httpx** | 前端调后端 |
| 可观测性 | **LangSmith** | 可选，env 一键开启 |

### 关键依赖版本

```
langchain>=1.3.0,<2.0.0          # 必须 1.x，API 与 0.x 完全不同
langchain-openai>=0.2.0
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
qdrant-client>=1.10.0
openai>=1.0.0
fastapi>=0.110.0
streamlit>=1.31.0
```

---

## 三、核心架构

### 3.1 Agent 决策循环

```
用户输入
   ↓
LangChain create_agent (CompiledStateGraph)
   ├── LLM 决策：闲聊？/ 调工具？
   │
   ├── 调 query_local_knowledge_base
   │       ↓
   │   SiliconFlow embedding ─► Qdrant 向量检索 ─► top-k chunks
   │       ↓
   │   LLM 基于 chunks 生成答案
   │
   ├── 调 read_entire_document
   │       ↓
   │   Qdrant scroll API 拉全部 chunks（按 chunk_index 排序还原）
   │       ↓
   │   LLM 基于全文生成答案
   │
   └── 不调工具：直接回答（闲聊 / 上下文引用）
```

**关键**：所有 tool_calls / tool_call_id 配对、消息顺序、状态机流转都由 `CompiledStateGraph` 内部托管，业务代码不再手动拼 messages。

### 3.2 双模式

| 模式 | 入口 | 行为 |
|---|---|---|
| **Agent 模式** | `RAGAgent.run()` / `.query_stream()` | LLM 决策是否调工具，闲聊不检索 |
| **Direct RAG 模式** | `DirectRAGPipeline.query_stream()` | 永远先检索后回答（MMR 重排） |

UI 侧栏可切换。

### 3.3 流式输出协议

```
# Streamlit 内联调用
for token in agent.query_stream(question):
    yield token   # st.write_stream 渲染

# FastAPI SSE
event: data\r\n
data: {"token": "..."}\r\n
\r\n
data: {"done": true}\r\n
\r\n
```

`query_stream` 内部实现是**两段式**：
1. `_graph.stream(stream_mode="messages")` 拉 token 给前端实时显示
2. 流结束后 `_graph.invoke()` 取**完整规范化历史**写回 `chat_history`，保证 tool_calls / ToolMessage 不丢

### 3.4 工具设计

| 工具 | 用途 | 实现 |
|---|---|---|
| `query_local_knowledge_base(query)` | 语义相似度检索 | SiliconFlow embed → Qdrant top-k |
| `read_entire_document(source_name)` | 按文件名拉全文 | Qdrant scroll + 按 `chunk_index` 重组 |

工具用 `@tool` 装饰器声明，函数 docstring 自动转 JSON Schema 喂给 LLM。

### 3.5 重试与降级

| 失败场景 | 策略 |
|---|---|
| Qdrant 网络抖动 / SSL 断 | `_retry()` 装饰器：4 次指数退避 (0.5/1/2/4s) |
| Streamlit `query_stream` 中途异常 | catch 后 `placeholder.error()` 显示，不污染历史 |
| LangChain agent 超过 `max_iterations` | 直接返回 "Agent exceeded reasoning limit" |
| 上传非支持类型 | FastAPI 400 + 明确 detail |

---

## 四、关键修复历史（值得记录的坑）

### 4.1 Qdrant 区域选错 → SSL EOF

`sa-east-1`（巴西）从国内访问跨太平洋 + 大西洋，TLS 握手中途断。**代码无解**，必须把集群迁到亚太区或自建。

### 4.2 `query_stream` 历史断层 → 下一轮 400

第一版用 `stream_mode="messages"` 拿 token 后手动拼 `[HumanMessage, finalAIMessage]`，**丢掉了中间的 `AIMessage(tool_calls)` 和 `ToolMessage`**。下一轮加载历史时 tool 消息变孤儿，DeepSeek 返回：

```
Messages with role 'tool' must be a response to a preceding message with 'tool_calls'
```

**修复**：流结束后再 `_graph.invoke()` 一次，把规范化的完整 messages 链写回 `chat_history`。

### 4.3 `config` 包名与 `.pyc` 缓存冲突

把 `config.py` 重构为 `config/` 包后，旧的 `__pycache__/config.cpython-312.pyc` 还在，Python 优先加载，导致 `ImportError`。**根治**：包改名 `app_config/`，彻底消除歧义。

### 4.4 LangChain 1.3 API 大改

`AgentExecutor` + `create_openai_tools_agent` 在 1.x 都没了，统一为 `create_agent(model, tools, system_prompt)` 返回 `CompiledStateGraph`。

### 4.5 Windows GBK 控制台 emoji 崩

DeepSeek 返回带 😊 的字符串，Windows 控制台 `UnicodeEncodeError`。`main.py` 启动时手动包 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`。

---

## 五、API 速查（FastAPI 后端）

| 端点 | 方法 | 用途 |
|---|---|---|
| `/health` | GET | 健康检查 + Qdrant 集合统计 |
| `/sources` | GET | 列出已索引文件名 |
| `/sources/{name}` | DELETE | 删除指定文件的所有 chunks |
| `/sessions/{sid}/clear` | POST | 清空某 session 的对话历史 |
| `/ingest` | POST (multipart) | 上传文档并索引 |
| `/chat` | POST | 同步问答 |
| `/chat/stream` | POST | SSE 流式问答 |
| `/docs` | GET | 自动生成的 OpenAPI 文档 |

**请求体规范**：
```json
{
  "session_id": "user-abc",
  "message": "硅基流动是什么？",
  "mode": "agent"   // 或 "direct"
}
```

---

## 六、数据流图

### 单机 Demo（Streamlit 直连）

```
┌──────────────────┐
│  Streamlit UI    │
│  app.py :8501    │
└────────┬─────────┘
         │ 进程内调用
         ▼
┌──────────────────┐
│  RAGAgent /      │
│  DirectRAGPipe   │
└────────┬─────────┘
         │
   ┌─────┼──────────┬──────────┐
   ▼     ▼          ▼          ▼
DeepSeek SiliconFlow Qdrant   Logs
 (云)    (云)        (云)
```

### 前后端分离（生产形态）

```
┌──────────────────┐       HTTP/SSE       ┌──────────────────┐
│  Streamlit UI    │ ─────────────────► │  FastAPI Backend │
│  app.py :8501    │ ◄───────────────── │  server.py :8000 │
└──────────────────┘                    └────────┬─────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                    ┌───────────┐         ┌──────────────┐      ┌──────────────┐
                    │ DeepSeek  │         │ SiliconFlow  │      │   Qdrant     │
                    │   LLM     │         │  Embedding   │      │    Cloud     │
                    │   (云)    │         │     (云)     │      │     (云)     │
                    └───────────┘         └──────────────┘      └──────────────┘
```

### Session 复用模型（后端内存）

```
session_id → RAGAgent (含 chat_history)
                 ├── 共享 EmbeddingService 单例（HTTP client）
                 └── 共享 VectorStore 单例（Qdrant client）
```

多用户隔离仅按 `session_id`，进程重启即清空（要持久化得接 Redis）。

---

## 七、扩展实施指南

### 7.1 LangSmith Tracing（小）

**改这些**

| 文件 | 改动 |
|---|---|
| `.env` | 加 `LANGCHAIN_TRACING_V2=true` / `LANGCHAIN_API_KEY=lsv2_...` / `LANGCHAIN_PROJECT=RAG_Agent` |
| 无需改代码 | `core/llm.py` 启动时已自动写 env vars，LangChain 自动捕获 |

**工程量**：5 分钟。

---

### 7.2 Web 搜索工具（小）

**改这些**

| 文件 | 改动 |
|---|---|
| `requirements.txt` | 加 `tavily-python` |
| `.env` | 加 `TAVILY_API_KEY=...` |
| `tools/registry.py` | 把 `search_web` 占位函数改成调 `TavilyClient().search(query)` |
| `core/prompt.py` | 在 `AGENT_SYSTEM_PROMPT` 里把 search_web 说清楚什么时候用 |
| `core/agent.py` | `RAGAgent(enable_web_search=True)` 默认开启 |

**工程量**：~50 行，1 小时。

---

### 7.3 LLM 模型切换（小）

DeepSeek 切到通义千问 / GPT / Claude：

| 文件 | 改动 |
|---|---|
| `app_config/__init__.py` | 改 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` |
| `core/llm.py` | 无需改 —— `build_llm()` 用 OpenAI 兼容协议，任何兼容 API 都能接 |

**Claude 例外**：得换成 `ChatAnthropic`。

---

### 7.4 多用户 + Redis Session（中）

**改这些**

| 文件 | 改动 |
|---|---|
| `requirements.txt` | 加 `redis` |
| 新增 `core/session_store.py` | 用 Redis 存 `session_id → 序列化后的 chat_history (LangChain dumps/loads)` |
| `server.py` | `get_agent()` 改为从 Redis 拉 → 反序列化构 RAGAgent → 用完写回 |

**工程量**：~100 行，1 天。

---

### 7.5 Docker Compose 部署（中）

**新增文件**

| 文件 | 作用 |
|---|---|
| `Dockerfile.backend` | 装 Python 依赖，跑 `uvicorn server:app` |
| `Dockerfile.frontend` | 装 Python 依赖，跑 `streamlit run app.py` |
| `docker-compose.yml` | 编排两个容器 + 共享网络 + 注入 `BACKEND_URL=http://backend:8000` |

**关键**：前端容器里 Streamlit 不再 import `RAGAgent`，改成走 httpx 调 `BACKEND_URL`。这部分代码还没改造完。

**工程量**：~150 行 + Dockerfile 调试，半天。

---

### 7.6 Ragas 自动评测（中）

**改这些**

| 文件 | 改动 |
|---|---|
| 新增 `evals/dataset.json` | 50-100 条 `{question, ground_truth}` |
| 新增 `evals/run_evals.py` | 跑 dataset → 调 `agent.run()` → 收集 contexts → ragas |
| `requirements.txt` | 加 `ragas` `datasets` |

**Ragas 配置**：
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ChatOpenAI(model=DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY),
    embeddings=...,
)
```

**工程量**：测试集是大头，代码本身半天。

---

### 7.7 OCR 扫描件 PDF（中）

**改这些**

| 文件 | 改动 |
|---|---|
| `requirements.txt` | 加 `paddleocr>=2.7` 或 `pytesseract` |
| `database/document_loader.py` | `load_pdf` 先 `extract_text()`，空字符串则用 pdfplumber 提图 + OCR |
| 新增 `database/ocr.py` | OCR 单独模块，`@lru_cache` 缓存初始化 |

**注意**：PaddleOCR 首次下 50MB 模型，**只在 pypdf 提取为空时兜底**，不要全量 OCR。

**工程量**：~150 行 + 模型下载，1-2 天。

---

### 7.8 多模态（图文混合索引）（大）

**改这些**

| 文件 | 改动 |
|---|---|
| `database/document_loader.py` | PDF 同时提取页面图，data URL 形式存 |
| `core/llm.py` | 加 vision LLM（`qwen-vl-max` 或 `glm-4v`） |
| 新增 `core/multimodal.py` | 图 → caption / OCR 文本 |
| `database/vector_store.py` | 新增 collection `rag_documents_images`，独立 embedding |
| `tools/registry.py` | 新工具 `search_images_by_description` |

**工程量**：~400 行 + vision 模型成本评估，3-5 天。

---

### 7.9 持久化 chat_history（小）

目前 Streamlit 直连模式刷新页面就丢历史。

| 文件 | 改动 |
|---|---|
| `app.py` | `chat_history` 写到本地 sqlite，按 `session_id` 索引 |
| 或者：直接复用上面 7.4 的 Redis 方案 | |

**工程量**：~30 行，半小时。

---

### 7.10 切换到自建 Qdrant（中）

**目标**：脱离 Qdrant Cloud（避免出境 / 节省成本 / 部署到国内）。

| 文件 | 改动 |
|---|---|
| `docker-compose.yml` | 加 `qdrant/qdrant` 容器，端口 6333 |
| `.env` | `QDRANT_URL=http://qdrant:6333`，删 `QDRANT_API_KEY`（本地不用） |
| `database/vector_store.py` | 无需改，client 接口一致 |

**工程量**：10 分钟（前提：本地装好 Docker）。

---

## 八、扩展优先级建议

按"投入 / 收益"排序：

1. **7.1 LangSmith Tracing** — 5 分钟，调试效率指数级提升
2. **7.10 切自建 Qdrant** — 10 分钟，免运维 + 免出境合规风险
3. **7.9 持久化 history** — 半小时，刷新不丢话
4. **7.3 LLM 切换** — 10 分钟，灵活换模型 A/B
5. **7.2 Web 搜索工具** — 1 小时，Agent 能力对称化
6. **7.5 Docker Compose** — 半天，正式部署的前提
7. **7.6 Ragas 评测** — 1 天，质量基线
8. **7.4 多用户 Redis** — 1 天，真正多用户场景
9. **7.7 OCR** — 1-2 天，扫描件刚需才做
10. **7.8 多模态** — 3-5 天，质变功能

---

## 九、扩展面（其他想到没想到的方向）

- **MCP 接入** —— 让本系统作为 MCP 服务端，Claude Desktop / Cline 等直接调
- **管理后台** —— 单独的 admin UI 查看 / 编辑 / 删除知识库内容
- **Slack / 飞书机器人** —— 把 `/chat` 端点接到 IM 平台
- **跨文档关系图** —— 用 LLM 抽实体，可视化文档间链接
- **HyDE（假设答案检索）** —— 复杂问题先让 LLM 写假想答案，再 embed 检索
- **子问题拆解** —— 长 / 多跳问题先拆成几个子问题分别检索
- **Re-ranker** —— Qdrant 召回 top-40 后过 CrossEncoder 重排取 top-15
- **混合检索** —— Qdrant 向量 + 关键词（Qdrant 1.10+ 支持稀疏向量 + BM42）
- **离线 LLM 兜底** —— 本地 Ollama qwen2.5:7b 作为 DeepSeek 失败时的降级
- **审计日志** —— 每次问答 + tool_calls + 检索 chunks 落到时序数据库

---

## 十、当前状态快照

| 组件 | 状态 |
|---|---|
| Streamlit UI（`app.py`） | ✅ 代码完整，运行依赖 Qdrant 可达 |
| FastAPI 后端（`server.py`） | ✅ 代码完整，运行依赖 Qdrant 可达 |
| LangChain Agent | ✅ 已验证 4 轮多类型对话通过 |
| SiliconFlow Embedding | ✅ 已验证 1024 维向量正确 |
| Qdrant Cloud | ⚠️ 当前集群在 `sa-east-1` 国内访问不稳，待迁移亚太或自建 |
| LangSmith Tracing | ✅ 接口就绪，env 一开即用 |
| 重试机制 | ✅ Qdrant 每个方法都包了 4 次指数退避 |
| Docker 部署 | ⏸️ 未做 |
| 多用户 / Redis | ⏸️ 未做 |

整套系统已经从单文件原型走到 **生产骨架就绪**：覆盖了路由判断、工具调用、检索、流式生成、多轮对话、文档增删、Streamlit + FastAPI 双形态。后续按上面 7 / 8 章节路线图推进即可。
