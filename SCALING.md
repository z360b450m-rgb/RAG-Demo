# 高并发改造路线图

> **这是什么**：把 RAG-Demo 从「10 并发玩具」一步步推到「10000 并发生产 SaaS」需要做的所有改造。
>
> **怎么用**：每一档分独立章节，按用户增长挑章节做。不需要一次性全做完。
>
> **现状（v1.0）**：单进程 Streamlit / 单 worker FastAPI / 内存 session dict，能撑 10-30 并发。

---

## 〇、现状压力测试

### 当前架构能撑多少（理论值）

| 组件 | 形态 | 单机理论上限 | 真实瓶颈 |
|------|------|--------------|---------|
| Streamlit UI | 单进程 | ~10 并发 | Streamlit 设计就不是高并发的 |
| FastAPI 后端 | 单 uvicorn worker | ~100 RPS | GIL 限制 + LLM 调用同步等待 |
| `SESSIONS` dict | 进程内存 | ~1000 session | 每个 RAGAgent ~10MB |
| DeepSeek API | HTTP | 200 RPM (默认套餐) | 撞限流就 429 |
| SiliconFlow Embedding | HTTP | ~1000 RPM (免费) | 撞限流就 429 |
| Qdrant Cloud Free | 1GB 集群 | ~50 QPS | 免费版限流 |

**综合**：~10-30 并发用户能稳定用，超过开始排队 / 超时 / OOM。

### 简易压测命令

```bash
# 装 hey
go install github.com/rakyll/hey@latest

# 测 50 并发持续 30 秒
hey -z 30s -c 50 -m POST \
  -H "Content-Type: application/json" \
  -d '{"session_id":"load","message":"你好","mode":"agent"}' \
  http://localhost:8000/chat
```

观察 P99 / 错误率 / 内存。这是你做任何改造前必跑的基线。

---

## 一、档位 1：100 并发（小团队 / 内测）

**目标**：单台中等服务器扛住，不丢请求。

**改造清单（按优先级）**：

### 1.1 多 Worker 进程（5 分钟）

**问题**：单 uvicorn worker 撞 GIL，CPU 利用不满 1 核。

**改**：

```bash
# 启动时加 --workers
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

# 或用 Gunicorn 管 uvicorn worker
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**注意**：Worker 之间内存不共享，立刻暴露下面的 SESSIONS 问题。

---

### 1.2 Session 移出内存（1 天）

**问题**：`server.py:SESSIONS = {}` 是进程内存 dict，多 Worker 之间不共享。同一用户连续两个请求被路由到不同 Worker → 对话历史断了。

**改**：

```python
# 新增 core/session_store.py
import json
from typing import Optional
import redis
from langchain_core.load import dumps, loads
from langchain_core.messages import BaseMessage


class SessionStore:
    def __init__(self, redis_url: str, ttl: int = 86400):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl

    def load_history(self, session_id: str) -> list[BaseMessage]:
        raw = self.r.get(f"session:{session_id}")
        if not raw:
            return []
        return [loads(s) for s in json.loads(raw)]

    def save_history(self, session_id: str, history: list[BaseMessage]):
        serialized = [dumps(m) for m in history]
        self.r.setex(f"session:{session_id}", self.ttl, json.dumps(serialized))

    def clear(self, session_id: str):
        self.r.delete(f"session:{session_id}")
```

`server.py` 改造：

```python
SESSION_STORE = SessionStore(os.getenv("REDIS_URL", "redis://localhost:6379"))

def get_agent(session_id: str) -> RAGAgent:
    # 不再缓存 agent，每次新建（agent 创建很快，只是 dict）
    agent = RAGAgent(embedding_service=EMB, vector_store=VEC)
    agent.chat_history = SESSION_STORE.load_history(session_id)
    return agent

@app.post("/chat")
def chat(req: ChatRequest):
    agent = get_agent(req.session_id)
    answer = agent.run(req.message)
    SESSION_STORE.save_history(req.session_id, agent.chat_history)
    return {"response": answer}
```

**额外好处**：Redis 还能承载下面的 embedding 缓存 / 限流计数 / 任务队列。

---

### 1.3 加请求超时（1 小时）

**问题**：DeepSeek 偶尔慢到 30 秒，HTTP 连接拖着 worker 不释放。

**改**：每个端点包 `asyncio.wait_for`，超时返回 504。

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        async with asyncio.timeout(60):
            agent = get_agent(req.session_id)
            answer = await asyncio.to_thread(agent.run, req.message)
            SESSION_STORE.save_history(req.session_id, agent.chat_history)
            return {"response": answer}
    except asyncio.TimeoutError:
        raise HTTPException(504, "LLM timeout")
```

`build_llm()` 也加 `timeout=30, max_retries=2`。

---

### 1.4 请求限流（2 小时）

**问题**：单用户疯狂刷请求会把整台机器拖垮。

**改**：`slowapi` 中间件，每用户每分钟 30 次。

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=lambda req: req.headers.get("X-Session-ID", get_remote_address(req)),
    storage_uri="redis://localhost:6379",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    ...
```

---

### 1.5 Streamlit 改前后端分离（半天）

**问题**：`app.py` 内嵌业务逻辑，Streamlit 单进程是真瓶颈。

**改**：

| 文件 | 改动 |
|------|------|
| `app.py` | 删掉所有 `RAGAgent()` / `DirectRAGPipeline()` 引用，全部改用 `httpx.post(BACKEND_URL + "/chat/stream", ...)` |
| 新增 `client.py` | 封装 httpx 调用 + SSE 解析 |

```python
# client.py
import httpx, json

class BackendClient:
    def __init__(self, base_url: str):
        self.base = base_url
        self.http = httpx.Client(trust_env=False, timeout=120)

    def chat_stream(self, session_id: str, message: str, mode: str = "agent"):
        with self.http.stream("POST", f"{self.base}/chat/stream",
                              json={"session_id": session_id, "message": message, "mode": mode}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    if "token" in evt:
                        yield evt["token"]
                    elif evt.get("done"):
                        break
```

`app.py` 关键改造：

```python
client = BackendClient(os.getenv("BACKEND_URL", "http://localhost:8000"))
session_id = st.session_state.get("sid") or st.session_state.setdefault("sid", str(uuid.uuid4()))

for token in client.chat_stream(session_id, question):
    st.write_stream(...)
```

---

### 1.6 Qdrant 套餐升级 / 自建

免费版 Qdrant Cloud 1GB / 限速 ~50 QPS，100 并发会撞。

**选项 A**：升 Qdrant Cloud Standard（按月，几十刀起）

**选项 B**：自建 Qdrant in Docker（推荐）

```yaml
# docker-compose.yml 加一段
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["./qdrant_storage:/qdrant/storage"]
```

`.env`：`QDRANT_URL=http://qdrant:6333`，删 `QDRANT_API_KEY`。

---

### 1.7 档位 1 完成后的样子

```
┌──────────────┐
│  Streamlit   │  (轻量纯前端)
│   :8501      │
└──────┬───────┘
       │ HTTP
       ▼
┌──────────────────────────────────┐
│  FastAPI Backend                 │
│  uvicorn --workers 4 :8000       │
│  + slowapi 限流                  │
│  + asyncio.timeout 超时          │
└──┬──────────────┬────────────────┘
   │              │
   ▼              ▼
┌──────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐
│  Redis   │  │ DeepSeek  │  │SiliconFl│  │ Qdrant   │
│ Session  │  │  LLM API  │  │  Embed  │  │ (自建)   │
│ Storage  │  │           │  │         │  │          │
└──────────┘  └───────────┘  └─────────┘  └──────────┘
```

**能撑 100-200 并发用户**，单机 4 核 8G 服务器 ~¥80/月（云上抢占式）。

---

## 二、档位 2：1000 并发（生产 SaaS）

在档位 1 基础上加：

### 2.1 后端横向扩容（K8s 或 Swarm）

把 FastAPI 容器化、复制 3-5 份，前面挂负载均衡。

**最小改动**：用 docker compose `--scale`：

```yaml
# docker-compose.yml
services:
  backend:
    build: .
    deploy:
      replicas: 5
    environment:
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
  
  nginx:
    image: nginx
    ports: ["8000:80"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf"]
    depends_on: [backend]
```

```nginx
# nginx.conf
upstream backend {
    server backend:8000;  # docker DNS 自动负载均衡
}
server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # SSE 需要 keepalive
        proxy_buffering off;             # SSE 不能缓冲
        proxy_read_timeout 300s;
    }
}
```

**真生产用 K8s**：`kubectl apply -f deployment.yaml`，HPA 按 CPU 自动扩缩。

---

### 2.2 Embedding 缓存（半天）

**问题**：同样的查询字符串重复 embed，浪费 API 配额。

**改**：Redis 缓存 `sha256(text) → vector[1024]`。

```python
# core/llm.py
import hashlib, json

class EmbeddingService:
    def __init__(self, redis_client=None):
        self.client = OpenAI(...)
        self.cache = redis_client  # 可选

    def embed_query(self, query: str) -> list[float]:
        if self.cache:
            key = f"emb:{hashlib.sha256(query.encode()).hexdigest()}"
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
        vec = self.client.embeddings.create(model=self.model, input=[query]).data[0].embedding
        if self.cache:
            self.cache.setex(key, 86400, json.dumps(vec))  # 1 天 TTL
        return vec
```

**收益**：高频问题命中率 30-50%，省同样比例的 SiliconFlow 配额。

---

### 2.3 语义查询缓存（1 天）

**问题**：用户问"硅基流动是什么"和"什么是 SiliconFlow"是同一个意图，但 embedding 不同，都跑完整 RAG 链路。

**改**：用 `langchain-redis` 的 `RedisSemanticCache`，相似度 > 0.95 直接返回历史答案。

```python
from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache

set_llm_cache(RedisSemanticCache(
    redis_url=os.getenv("REDIS_URL"),
    embedding=...,  # 复用我们的 EmbeddingService 包装
    distance_threshold=0.05,  # 1 - similarity 阈值
))
```

**注意**：
- 同样的 prompt 模板才会命中（chat_history 不同 → 不命中）
- 实时性强的问题别缓存（用户问"现在几点了"）

---

### 2.4 慢操作异步化（2 天）

**问题**：`POST /ingest` 上传 100MB PDF 同步处理，worker 卡 60 秒。

**改**：Celery + Redis 后台跑。

```python
# tasks.py
from celery import Celery
celery = Celery("rag", broker=REDIS_URL, backend=REDIS_URL)

@celery.task
def ingest_async(file_path: str) -> dict:
    pipeline = DirectRAGPipeline()
    count = pipeline.ingest_document(Path(file_path))
    return {"chunks_added": count}
```

```python
# server.py
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    fp = save_to_tmp(file)
    task = ingest_async.delay(str(fp))
    return {"task_id": task.id, "status": "queued"}

@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    result = ingest_async.AsyncResult(task_id)
    return {"state": result.state, "result": result.result if result.ready() else None}
```

前端轮询 `/tasks/{id}` 看进度。

---

### 2.5 熔断与降级（1 天）

**问题**：DeepSeek 挂 10 秒，所有请求堆积 → 雪崩。

**改**：`pybreaker` 给三个外部依赖各加一个熔断器。

```python
import pybreaker

llm_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
embed_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
qdrant_breaker = pybreaker.CircuitBreaker(fail_max=10, reset_timeout=30)

@llm_breaker
def call_llm(messages):
    return llm.invoke(messages)
```

熔断打开后直接返回 503 + Retry-After 头，不让请求堆积。

---

### 2.6 监控（1 天）

**问题**：不知道什么时候快崩了。

**改**：Prometheus + Grafana。

```python
# 装 prometheus-fastapi-instrumentator
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

`/metrics` 端点自动暴露每个 endpoint 的 QPS / P50/P95/P99 / 错误率。Grafana 配仪表盘看趋势。

**关键告警阈值**：
- P99 > 30s → 钉钉报警
- 5xx 错误率 > 1% → 钉钉
- Redis 内存 > 80% → 邮件
- DeepSeek 429 错误 > 100/min → 立即扩容或申请提额

---

### 2.7 多模型路由（2 天，看预算）

**问题**：所有问题都用 DeepSeek，简单 query 性价比低。

**改**：路由策略

| 问题类型 | 路由到 | 成本 |
|----------|--------|------|
| 闲聊 / 短问候 | 不调 LLM，规则回答 | 0 |
| 简单事实 | DeepSeek-V3（默认） | 低 |
| 复杂推理 / 多跳 | DeepSeek-R1 | 中 |
| 代码 / 数学 | Qwen-coder | 中 |

```python
# core/llm.py 加路由层
def route_llm(question_type: str) -> ChatOpenAI:
    if question_type == "simple": return build_llm(model="deepseek-chat")
    if question_type == "complex": return build_llm(model="deepseek-reasoner")
    return build_llm()
```

判定 `question_type` 用一个小模型或简单 heuristic（长度 / 关键词）。

---

### 2.8 档位 2 完成后的样子

```
                     ┌──────────────┐
                     │   CDN / WAF  │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │   Nginx LB   │
                     └──┬───────────┘
                        │ HTTP / SSE
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐
   │FastAPI │      │FastAPI │ ...  │FastAPI │ × 5 Pod
   │Worker  │      │Worker  │      │Worker  │
   │+ breaker│     │+ breaker│     │+ breaker│
   └───┬────┘      └───┬────┘      └───┬────┘
       │               │               │
       └───────┬───────┴───────┬───────┘
               ▼               ▼
         ┌──────────┐    ┌──────────────┐
         │  Redis   │    │   Celery     │
         │ Cluster  │    │  Workers     │
         │ Session  │    │  (ingest)    │
         │ + Cache  │    └──────────────┘
         └──────────┘
                                ▼
              ┌───────────┐  ┌──────────────┐  ┌──────────┐
              │ DeepSeek  │  │ SiliconFlow  │  │ Qdrant   │
              │  企业版   │  │  企业版      │  │ 自建集群 │
              │  ¥¥¥     │  │   ¥¥¥       │  │  3 节点  │
              └───────────┘  └──────────────┘  └──────────┘
                       │              │             │
                       └──────┬───────┴─────────────┘
                              ▼
                       ┌──────────────┐
                       │ Prometheus + │
                       │  Grafana     │
                       └──────────────┘
```

**能撑 1000-3000 并发用户**，月成本 ¥2000-5000（云费 + LLM token + 监控）。

---

## 三、档位 3：10000+ 并发（大规模生产）

档位 2 基础上加：

### 3.1 多区域部署

国内用户 / 海外用户路由到最近的区域，Qdrant 也按区域分集群。

**架构**：
- 北京 + 香港 + 新加坡三套独立部署
- DNS GeoIP 分流
- 跨区域只同步用户元数据（不同步对话历史）

---

### 3.2 向量库分片

单 Qdrant 集群撑不住 10000 QPS。按租户 / 时间 / 类目分多 collection。

**策略**：
- 多租户：每个企业客户独立 collection（`rag_<tenant_id>`）
- 时间分片：超过 30 天的 chunks 进 `rag_archive`，搜索时按需查归档

---

### 3.3 冷热数据分层

**问题**：1 年内累计 1000 万 chunks，向量库存储 / 检索都慢。

**改**：
- 热数据（30 天内）：Qdrant 主集群
- 冷数据（30 天-1 年）：Qdrant 低配集群 / 阿里云 OSS + 自建检索
- 归档（>1 年）：纯 OSS，需要时再加载

---

### 3.4 大文件流式入库

**问题**：1GB PDF 一次性加载内存爆。

**改**：边解析边 embed，每 100 chunk 一批写入。

```python
def stream_ingest(file_path: Path):
    reader = PdfReader(str(file_path))
    batch_text, batch_meta = [], []
    for page_idx, page in enumerate(reader.pages):
        for chunk in chunker.chunk_text(page.extract_text(), source_path=file_path.name):
            chunk.metadata["page"] = page_idx
            batch_text.append(chunk.text)
            batch_meta.append(chunk.metadata)
            if len(batch_text) >= 100:
                yield embed_and_upsert(batch_text, batch_meta)
                batch_text, batch_meta = [], []
    if batch_text:
        yield embed_and_upsert(batch_text, batch_meta)
```

Celery 跑这个生成器，每 100 chunk 更新一次进度。

---

### 3.5 Sticky Session

如果 chat_history 还在 Redis（不在内存），不需要 sticky。但流式 SSE 连接如果中断重连，路由到不同 Pod 会丢上下文。

**改**：Nginx 用 `ip_hash` 或 cookie 粘性。

```nginx
upstream backend {
    ip_hash;  # 同一 IP 总路由到同一 Pod
    server backend1:8000;
    server backend2:8000;
    ...
}
```

---

### 3.6 DDoS 防护

阿里云 / 七牛 / Cloudflare 的 WAF + CC 防护。每个 IP 每分钟 60 次以上的请求自动 ban 5 分钟。

---

## 四、关键原则

不管在哪个档位，三条原则不变：

### 4.1 Stateless API

后端任何一个 Pod 重启 / 扩容 / 缩容都不能掉数据。

| 反例 | 正例 |
|------|------|
| `SESSIONS: dict = {}` 内存字典 | Redis 持久化 |
| 实例属性 `self.chat_history` | 方法参数 `history: list` |
| 本地文件锁 | Redis distributed lock |
| 单例 ML 模型 | 共享只读资源 OK，但不能写状态 |

---

### 4.2 慢操作异步化

任何超过 1 秒的同步操作都该改成：
- **流式**（SSE / WebSocket）—— 用户感知"在动"
- **任务队列**（Celery）—— 用户拿 task_id 后轮询

**当前哪些是慢同步**：
- `agent.run()` 整个 RAG 链路 5-15 秒 ← 改 SSE（已做）
- `pipeline.ingest_document()` 大 PDF 30-120 秒 ← 改 Celery
- `vector_store.list_sources()` 全量 scroll ← 加 Redis 缓存

---

### 4.3 优雅降级

外部依赖会挂，提前想好降级路径：

| 依赖 | 挂了怎么办 |
|------|-----------|
| DeepSeek API | 切到 Qwen / GPT 备用 |
| SiliconFlow Embedding | 缓存命中 + 队列等待 |
| Qdrant | 返回"知识库暂时不可用"，让 LLM 凭通用知识答 |
| Redis | 会话掉了重新开始，前端提示重连 |
| 后端集群整体挂 | 静态 HTML 兜底"系统维护中" |

---

## 五、压测与容量规划

### 5.1 关键指标

| 指标 | 健康值 | 警戒值 | 红线 |
|------|--------|--------|------|
| API P99 延迟 | < 5s | 5-15s | > 30s |
| 5xx 错误率 | < 0.1% | 0.1-1% | > 1% |
| LLM token 用量 | 按预算线性 | 突增 50% | 突增 200% |
| Worker CPU | < 50% | 50-80% | > 80% |
| Redis 内存 | < 60% | 60-80% | > 80% |
| Qdrant QPS | 套餐 50% | 70% | 90% |

---

### 5.2 压测脚本

```bash
# locust 压测
pip install locust
```

```python
# locustfile.py
from locust import HttpUser, task, between
import uuid, random

class RAGUser(HttpUser):
    wait_time = between(2, 8)  # 真实用户停顿

    def on_start(self):
        self.session_id = str(uuid.uuid4())
        self.questions = [
            "硅基流动是什么？",
            "Qdrant 怎么用？",
            "总结一下知识库",
            "你好",
            "DeepSeek 和 GPT 有什么区别",
        ]

    @task(10)
    def chat(self):
        self.client.post("/chat", json={
            "session_id": self.session_id,
            "message": random.choice(self.questions),
            "mode": "agent",
        })

    @task(1)
    def list_sources(self):
        self.client.get("/sources")
```

跑：

```bash
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10
```

打开 http://localhost:8089 看仪表盘。

---

### 5.3 容量规划公式（粗算）

**单 Worker 能力**：约 1 个 LLM 请求 / 5 秒 = 12 RPS（流式可以高一点，因为 worker 不阻塞）

**100 并发用户，每人每分钟 2 个问题** = 200/60 ≈ 3.3 RPS
→ 1 个 Worker 够，留 2 倍 buffer = 2 Worker

**1000 并发** = 33 RPS → 5 Worker
**10000 并发** = 333 RPS → 30 Worker，必须 K8s 自动扩容

---

## 六、成本估算（粗略，按 2026 年价格）

| 档位 | 月成本 | 说明 |
|------|--------|------|
| **学习 / Demo** | ¥0-100 | DeepSeek 实际用量 + SiliconFlow 免费 + Qdrant 免费 |
| **档位 1（100 并发）** | ¥100-500 | + 1 台云 ECS + Redis |
| **档位 2（1000 并发）** | ¥2000-8000 | + K8s 集群 + Qdrant Standard + 监控 |
| **档位 3（10000 并发）** | ¥20000+ | + 多区域 + WAF + 企业版 LLM 套餐 |

LLM token 成本另算，按 DeepSeek 估：
- 平均一次对话 2k token in + 1k token out = ¥0.005 / 次
- 1000 用户 / 天 / 5 次 = 5000 次 = ¥25/天 ≈ ¥750/月

---

## 七、按需挑选指南

> **不要一次性全做**，按你真实遇到的瓶颈挑：

| 你遇到的问题 | 看哪一节 |
|-------------|---------|
| 服务器 CPU 经常打满 | 1.1 多 Worker |
| 多 Worker 后用户对话错乱 | 1.2 Session 移 Redis |
| 长请求卡住其他用户 | 1.3 超时 |
| 单用户疯狂刷请求 | 1.4 限流 |
| Streamlit 卡 | 1.5 前后端分离 |
| Qdrant Cloud 撞限流 | 1.6 自建 / 升套餐 |
| 1 台机器扛不住 | 2.1 横向扩容 |
| SiliconFlow 用量爆 | 2.2 Embedding 缓存 |
| 同样问题重复算 | 2.3 语义缓存 |
| 上传 PDF 卡死 | 2.4 异步队列 |
| DeepSeek 挂导致全站 5xx | 2.5 熔断 |
| 不知道哪里慢 | 2.6 监控 |
| LLM 账单爆炸 | 2.7 多模型路由 |
| 一个机房不够 | 3.1 多区域 |
| 单 Qdrant 撑不住 | 3.2 分片 |
| 老数据拖慢检索 | 3.3 冷热分层 |
| GB 级文件入库爆内存 | 3.4 流式入库 |
| SSE 重连丢上下文 | 3.5 Sticky Session |
| 被刷 / 被攻击 | 3.6 DDoS 防护 |

---

## 八、当前代码到生产的差距清单

把架构差距收敛到具体文件：

| 文件 | 现状 | 高并发需改成 |
|------|------|-------------|
| `server.py:39 SESSIONS = {}` | 内存 dict | Redis SessionStore |
| `server.py:46 RAGAgent(...)` 每 session 新建 | OK 但每次重建 | 配 Redis 后变成"每请求新建 + 加载历史" |
| `server.py /ingest` 同步 | 阻塞 worker | Celery 异步 |
| `core/agent.py:81 chat_history` 实例属性 | OK | 改成方法参数（解锁多用户共享 agent） |
| `core/llm.py EmbeddingService` 单例 | OK | 加 Redis cache 装饰 |
| `core/llm.py build_llm()` | 无 timeout | 加 timeout=30 |
| `database/vector_store.py` 单例 | OK | 加熔断 |
| `app.py` 直接 import agent | 单 Streamlit 进程 | 改 BackendClient HTTP 调用 |
| 全局 | 无监控 | 加 prometheus-fastapi-instrumentator |
| 全局 | 无限流 | 加 slowapi |
| 全局 | 单机部署 | docker-compose + nginx LB |

---

> 这份路线图按需 cherry-pick，**不要把它当成立刻要做的 TODO list**。
> 等你真的有用户、真的撞墙、真的看到数据再做对应章节。提前做就是过度工程。
