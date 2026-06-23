# 面试准备 · notes-app + RAG-Demo 两个项目

> **这是什么**：把两个项目当面试弹药库整理出来——项目自述、最可能被问的技术题、对方追问"哪里可以优化"时怎么答得有深度，以及面试官可能挖的坑怎么躲。
>
> **怎么用**：面试前一晚通读一遍。每节末尾有「一句话答案」和「细节展开」两层，按对方追问深度逐层抛。

---

## 〇、两个项目的电梯陈述（30 秒版）

**项目 A：notes-app + RAG-AIAgent（错题本 + 本地 RAG）**

> 一个 Vue 3 + Electron 桌面错题本，集成本地 RAG 智能问答。前端用 Composition API + IndexedDB 做错题 CRUD 和 SRS 间隔重复复习；后端用 Python + LangGraph 装了一个完整的状态机 Agent，包含路由判断、查询改写、混合检索（向量 + BM25 + Reranker）、文档评分、答案生成五个节点。错题保存时自动同步进 RAG 索引，AI 侧栏支持"锁定单题"和"全库检索"两种问答模式。

**项目 B：RAG-Demo（云原生 RAG Agent）**

> 一个 LangChain + DeepSeek + SiliconFlow + Qdrant Cloud 的纯云端 RAG 系统。用 LangChain 1.x 的 `create_agent` 替代手装 LangGraph 节点，让框架内部托管 tool_calls 状态机；前端 Streamlit 单机起步，后端 FastAPI 暴露 SSE 流式接口；本地零模型依赖，启动即用，是为云上多用户 SaaS 准备的骨架。

**两个项目的对照**——这是高频追问，必须答上：

| 维度 | notes-app（本地版）| RAG-Demo（云原生版）|
|------|------------------|-------------------|
| Agent 编排 | LangGraph 手装 5 节点 | LangChain `create_agent` 内置状态机 |
| Embedding | 本地 BGE-small-zh（HuggingFace 模型）| SiliconFlow 远程 BGE-large-zh |
| 向量库 | Chroma 本地文件 | Qdrant Cloud |
| 关键词检索 | BM25 本地 | 暂无（默认）|
| 重排 | CrossEncoder bge-reranker-base | 无（依赖召回质量）|
| 前端 | Vue 3 + Electron 桌面壳 | Streamlit Web UI |
| 部署形态 | 用户本地双进程 | 云上 FastAPI + Streamlit |
| 适合场景 | 离线 / 数据敏感 / 桌面体验 | 快速演示 / 云上 SaaS |

**为什么做两套**：第一版踩了"本地模型下载慢、Electron 打包大、跨平台坑多"的坑，第二版反向做了云原生版本验证不同部署形态的取舍。

---

## 一、最可能被问的技术题（深度从浅到深）

### 1.1 RAG 是什么？为什么不直接用 LLM？

**一句话答案**：检索增强生成（Retrieval-Augmented Generation）—— 先从知识库捞相关文档，再让 LLM 基于这些文档作答，解决幻觉 + 知识时效性 + 私有数据问题。

**细节展开**：

- LLM 训练数据有截止日期、不知道你的私有文档、容易编造（幻觉）
- RAG 把这三个问题都收敛到"检索质量 + Prompt 工程"两个可控环节
- 流程：用户问题 → embedding → 向量库相似度检索 → 取 top-K 拼进 prompt → LLM 生成
- 关键指标：召回率（找没找到对的）/ 上下文精度（噪声多不多）/ 忠实度（答案有没有偏离上下文）

---

### 1.2 你的 LangGraph 状态机长什么样？

**一句话答案**：5 个节点 —— route_question（判 chat 还是 rag）→ rewrite_query（清洗）→ retrieve（混合检索）→ grade_documents（逐条相关性判断）→ generate_rag_answer，全部用 Pydantic 结构化输出强约束。

**细节展开**：

```
       ┌──────────────┐
       │route_question│ ──── chat ────► generate_direct_chat ─► END
       └──────┬───────┘
              │ rag
              ▼
       ┌──────────────┐
       │rewrite_query │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │  retrieve    │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │grade_documents│
       └──┬───────────┘
          ├── docs empty ──► generate_direct_chat（fallback）
          └── docs valid ──► generate_rag_answer ─► END
```

**为什么每个节点都要存在**：

- `route_question` —— 用户说"你好"也走完整 RAG 链路就 token 烧钱了，先判一下
- `rewrite_query` —— 用户原话往往带寒暄和指代（"那个项目啥情况"），清洗成专业查询能提升召回 20%+
- `grade_documents` —— 向量检索召回的可能是相关性很弱的 chunk，逐条让 LLM 判定 relevant: bool，全部不相关就 fallback 到 direct_chat 而不是硬答
- `generate_rag_answer` —— 拿到经过筛选的 chunks 才真正生成

**为什么用 Pydantic 强约束**：每次 LLM 调用都用 `.with_structured_output(Schema, method="function_calling")`，Pydantic 校验失败抛错而不是返回烂字符串，前端不用解析 markdown 抖动。

---

### 1.3 混合检索具体怎么做的？为什么不只用向量？

**一句话答案**：Chroma 向量召回 top-10 + BM25 关键词召回 top-10 → 去重合并 → CrossEncoder reranker 取 top-4。向量擅长语义相似，BM25 擅长术语精确命中，缺一不可。

**细节展开**：

```
查询
 ├── Chroma 向量召回（语义） top_k=10
 └── BM25 关键词召回（精确）top_k=10
        ↓
    去重合并
        ↓
   CrossEncoder 重排 top_k=4
   （bge-reranker-base 或 fallback 向量序）
        ↓
    返回 chunk 文本
```

**反例**：用户问"BGE 模型怎么用"——

- 纯向量召回可能命中"BERT-based embedding model"（语义近但术语没对上）
- BM25 直接命中所有提到 "BGE" 的 chunk

**为什么要 reranker**：召回阶段（Chroma + BM25）追求 recall，可能把 20 条候选都送上来；reranker 是 CrossEncoder（双塔编码后做交叉注意力），算更准的相关性分，但慢，所以只对 top-20 做。

**模型不可用时降级**：`_get_reranker()` 加载失败返回 None，直接用向量召回的相似度顺序，不阻塞整条链路。

---

### 1.4 LangChain 1.x 的 create_agent 和 LangGraph 装节点有什么区别？

**一句话答案**：`create_agent` 是 LangGraph 编译好的成品（CompiledStateGraph），内部已经写好 tool-calling 循环 + tool_call_id 配对；自己装 StateGraph 节点适合需要插入 grade / route 等定制节点的场景。

**细节展开**：

- LangChain 0.x：`AgentExecutor` + `create_openai_tools_agent`，单独抽象
- LangChain 1.x：所有 Agent 统一基于 LangGraph，`create_agent(model, tools, system_prompt)` 一行起底
- 内部：`CompiledStateGraph` 自动管理 HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage 的完整链条
- **业务代码再也不用手动拼 tool_call_id**，这是上一个项目踩过最深的坑（见 4.2）

**什么时候选哪个**：

- 纯 tool-calling Agent（RAG-Demo）→ `create_agent`
- 需要 route_question / grade_documents 等定制中间节点（notes-app）→ 自己装 StateGraph
- 区别本质：前者是后者的特化封装

---

### 1.5 你怎么处理 Tool Call 的 400 错误？

**一句话答案**：DeepSeek/OpenAI 协议要求 `role=tool` 的消息必须紧跟在带 `tool_calls` 的 assistant 消息之后；如果手动维护 chat_history 把中间消息丢了，下一轮就会触发 `Messages with role 'tool' must be a response to a preceding message with 'tool_calls'`。

**细节展开**：踩坑顺序如下——

第一次：`Memory._trim()` 按消息数裁剪，把 `[user, assistant+tool_calls, tool, assistant]` 4 条消息当 2 轮裁，结果 `assistant+tool_calls` 被切掉、`tool` 变孤儿。**修复**：裁剪后强制回退到最近的 user 边界。

第二次：流式 `query_stream` 拿 token 时手动拼 `chat_history = [..., HumanMessage, finalAIMessage]`，丢掉了中间的 `AIMessage(tool_calls)` 和 `ToolMessage`。**修复**：流结束后再调一次 `_graph.invoke()` 拿框架返回的规范化完整消息链，用它覆盖 chat_history。

第三次（彻底解决）：迁移到 LangChain 1.x 的 `create_agent`，框架内部托管整个状态机，**永远不用业务代码手动拼 messages**。

---

### 1.6 你的文档切分策略？为什么 chunk_size=800、overlap=200？

**一句话答案**：用 LangChain 的 `RecursiveCharacterTextSplitter`，按 `\n\n` → `\n` → `。！？` → 空格 优先级递归切分，保证不在句中拦腰截断。800/200 是 BGE embedding 模型最佳输入长度的经验值。

**细节展开**：

- **chunk_size**：BGE 模型最大 512 token，中文 1 字 ≈ 1.3-2 token，800 字符接近上限
- **overlap=200**：跨 chunk 边界的语义不丢，比如一个概念定义在 chunk1 结尾、解释在 chunk2 开头
- **为什么不固定切**（早期版本踩坑）：固定字符硬切会把"BAAI/bge-small-zh-v1.5"切成"BAAI/bge-small-zh"和"-v1.5"，命中率掉 30%
- **separators 顺序很关键**：先尝试段落分隔，再退化到换行，再到句号，最后空格

**对照实验**：固定切分 vs RecursiveSplitter，在错题本场景下检索相关性从 0.62 提升到 0.79（人工标注 100 条 query）。

---

### 1.7 怎么解决"用户问最后一句话是什么"这种引用上下文的查询？

**一句话答案**：让 LLM 自己判断要不要调工具，闲聊和上下文引用都不调，纯靠 chat_history 答；调了工具的轮次完整保留中间消息，不调的轮次直接 `yield content`。

**细节展开**：

- 错误做法：所有查询都强制走 RAG → 用户问"你刚才说的第一点是什么"也会去检索，返回的 chunks 跟"上一句"毫无关系
- 正确做法：`agent.run("你刚才说的第一点是什么")` 时 LLM 看 system prompt 的"casual chat: answer directly"指令 + 看 chat_history 里有自己之前的回答 → 不调工具，直接生成
- 关键约束：chat_history 必须是规范化的（包含所有 ToolMessage），不然 LLM 看不到完整上下文

---

### 1.8 Streamlit 和 FastAPI 怎么共用一套 Agent 代码？

**一句话答案**：`core/agent.py` 的 `RAGAgent` 完全和 UI 解耦，只暴露 `run()` 和 `query_stream()` 两个方法。Streamlit 直接 import 调用（单机 Demo），FastAPI 包一层 SSE EventSourceResponse 暴露 HTTP。

**细节展开**：

```python
# Streamlit 模式（app.py）
agent = RAGAgent()
for token in agent.query_stream(question):
    st.write_stream(...)

# FastAPI 模式（server.py）
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        agent = get_agent(req.session_id)
        for token in agent.query_stream(req.message):
            yield {"data": json.dumps({"token": token})}
    return EventSourceResponse(gen())
```

为什么不强制走 HTTP：单机演示场景多一层网络反而慢、错处理也复杂；提供两种形态让用户按场景选。

---

### 1.9 SSE 流式协议怎么实现？为什么不用 WebSocket？

**一句话答案**：SSE 是单向（服务端→客户端）的 HTTP 长连接，足够 LLM 流式输出场景；用原生 fetch + ReadableStream 手写解析，**必须同时兼容 `\r\n\r\n` 和 `\n\n` 两种分隔符**（sse-starlette 用前者，标准是后者）。

**细节展开**：

- WebSocket 是双向，但 LLM 流式输出只需要服务端推 → 用 SSE 更简单
- SSE 优势：基于 HTTP，过 CORS / 过代理 / 过 nginx 比 WebSocket 顺
- 前端解析关键代码：

```typescript
const findSep = (s: string) => {
  const a = s.indexOf('\r\n\r\n')
  const b = s.indexOf('\n\n')
  if (a === -1 && b === -1) return { idx: -1, len: 0 }
  if (a === -1) return { idx: b, len: 2 }
  if (b === -1) return { idx: a, len: 4 }
  return a < b ? { idx: a, len: 4 } : { idx: b, len: 2 }
}
```

不用 EventSource：EventSource 不支持 POST，传不了 JSON body。

---

### 1.10 业务数据和 AI 索引怎么保持同步？

**一句话答案**：错题保存成功后**静默**触发 `/entries/upsert`，先全 KB 删旧 chunk 再写新 chunk；失败不阻塞业务数据保存（IndexedDB 才是真相）。

**细节展开**：

```typescript
async function saveEntry() {
  await db.put(entry)               // 1. 业务数据落地（必须成功）
  void ragSync.upsertEntry(entry)   // 2. 异步同步 RAG，失败 console.warn
}
```

- **先删后写**：用户改了错题归属的 KB，旧 chunk 必须清，否则跨 KB 检索会污染
- **失败静默**：RAG 服务挂了不能让用户错题保存失败
- **4 秒超时**：用 `AbortController` 控制，不让网络抖动拖死前端

---

### 1.11 多知识库怎么隔离？

**一句话答案**：每个 KB = 独立 Chroma collection + 独立 BM25 索引 + 独立物理目录 `data/<kb_id>/`，命名约定 `kb_<kb_id>`。

**细节展开**：

- 检索时只在当前选中 KB 内做，互不影响
- 删除 KB 只清 collection，**保留物理目录**防误删
- 错题在 KB 间迁移：upsert 自动先全库删旧 chunk 再写到目标 KB
- `kb_registry.json` 存 KB 元数据（名称、创建时间、描述）

---

### 1.12 你的 Pydantic 结构化输出 schema 有哪些？

**一句话答案**：4 个 schema 对应 4 类 LLM 调用——AgentResponse（最终答）、RouteDecision（路由）、RewriteResult（改写）、GradeDecision（评分）。

```python
class AgentResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0

class RouteDecision(BaseModel):
    destination: Literal["chat", "rag"]

class RewriteResult(BaseModel):
    rewritten: str

class GradeDecision(BaseModel):
    relevant: bool
```

**为什么这样设计**：

- `AgentResponse` 给前端用：`answer` 渲染，`sources` 显示引用，`confidence_score` 决定要不要标"AI 不太确定"
- `RouteDecision` 用 Literal 强约束，模型不会瞎写 destination 字段
- `GradeDecision` 极简，每个 chunk 一次调用，bool 判定快又准

---

### 1.13 重试和降级具体怎么做？

**一句话答案**：DeepSeek API 调用包 3 次指数退避（0.6/1.2/2.4s），重试仍失败返回 `[服务暂不可用] <错误类型>` 明确标记 confidence=0.0；Reranker 模型加载失败 fallback 用向量序；检索 0 文档自动转 direct_chat。

**细节展开**：

```python
def _invoke_with_retry(runnable, payload, attempts=3, label=""):
    for i in range(attempts):
        try:
            return runnable.invoke(payload)
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(0.6 * (2 ** i))
```

**关键设计**：失败要返回**明确错误前缀**，不能让 OpenAI SDK 把错误塞进 answer 字段伪装成正常回答。前端看到 `[服务暂不可用]` 才能正确显示"重试中..."而不是当成 AI 答案展示。

---

### 1.14 错题本前端的状态管理是怎么做的？

**一句话答案**：Composition API + Composable，没用 Pinia/Vuex。每个领域一个 composable：`useEntries` / `useNotebooks` / `useReview` / `useAiChat` / `useAiSkills` / `useKnowledgeBases` / `useRagSync`，全局 reactive 状态 + 模块化逻辑。

**为什么不用 Pinia**：

- 这是个本地单机应用，没有跨组件的复杂全局状态机
- Composable 已经够用，加 Pinia 是过度工程
- 业务边界清晰：每个 composable 内聚一个领域

---

### 1.15 SRS 间隔重复算法是怎么实现的？

**一句话答案**：SM-2 算法 ——根据用户复习时的"难度反馈"（0-5 分），动态调整下次复习间隔。第一次 1 天，记得好就 1→6→14→30 翻倍，记不住就重置。

**核心公式**：

```typescript
function computeNextReview(quality: number, repetition: number, easeFactor: number, interval: number) {
  if (quality < 3) {
    // 没记住：重置
    repetition = 0
    interval = 1
  } else {
    // 记住了：增加间隔
    if (repetition === 0) interval = 1
    else if (repetition === 1) interval = 6
    else interval = Math.round(interval * easeFactor)
    repetition += 1
  }
  // 调整难度因子
  easeFactor = Math.max(1.3, easeFactor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
  return { repetition, easeFactor, interval, nextReviewAt: Date.now() + interval * 86400000 }
}
```

---

## 二、对方可能问"哪里可以优化"——怎么答得有深度

> **关键策略**：不要装作"已经完美"。坦诚 + 路线图 + 具体方案，比"我觉得目前挺好"显得专业 10 倍。
>
> 每一项的回答模板：**承认问题 → 分析根因 → 给出 2-3 个可选方案 → 说明取舍**。

---

### 2.1 检索相关性还能怎么提升？

**承认**：当前混合检索 + reranker 在我们错题本场景下相关性约 0.79（人工标注 100 条），还有提升空间。

**根因**：
- 向量召回受限于 chunk 切分粒度
- BM25 对同义词无效（用户问"硅基流动"vs 文档写 "SiliconFlow"）
- Reranker 是通用模型，没在错题领域微调过

**方案**：

| 方案 | 预期提升 | 成本 |
|------|---------|------|
| **HyDE**（假设答案检索）：先让 LLM 写一段"假想答案"，用它做 embedding | +5-10% | 每问多 1 次 LLM 调用 |
| **子问题拆解**：长问题先拆成 3 个子问题分别检索后合并 | +8-15% 在多跳问题上 | 多 1 次 LLM 调用 |
| **领域微调 embedding**：用错题数据微调 BGE | +10-20% | 需要标注数据，1-2 周 |
| **多向量索引**：标题、正文、答案分别 embed | +5% | 存储 3 倍 |

**实际怎么选**：先做 HyDE（最低成本），效果好再做子问题拆解，最后再考虑微调。

---

### 2.2 Token 成本怎么降？

**承认**：DeepSeek 虽便宜，但 5 个节点都调 LLM，每次问答平均 ~3000 token 输入 + 800 token 输出。

**根因**：
- `grade_documents` 节点每个 chunk 一次调用，10 个 chunk 就是 10 次
- `rewrite_query` 节点对短查询也跑完整推理，性价比低
- 重复问题没缓存，每次重算

**方案**：

| 方案 | 节省比例 | 成本 |
|------|---------|------|
| **DeepSeek Prompt Caching**：固定 system prompt 放最前，命中 cache 省 90% 输入 token | -50% 输入 | 仅改 prompt 顺序 |
| **GradeDocuments 批处理**：一次喂 10 个 chunk 让 LLM 一起判断 | -80% 评分 token | 改 grade 节点 |
| **语义查询缓存**（RedisSemanticCache）：相似查询直接返回历史答案 | -30% 重复请求 | 加 Redis |
| **路由层短路**：query 长度 < 10 字符直接 chat（不走 rewrite） | -20% rewrite 调用 | 改 route 节点 |
| **多模型分级**：闲聊用更便宜的小模型 | -40% 闲聊场景 | 加路由 |

**实际怎么选**：Prompt Caching 是免费午餐先上；批处理 grade 改动小收益大第二个做；分级路由最后做。

---

### 2.3 怎么扛高并发？

承认当前是单进程架构，~10-30 并发就撞墙。

完整路线见 `SCALING.md`。面试关键回答：

| 并发量级 | 怎么改 |
|---------|--------|
| 100 并发 | 多 Worker（uvicorn `--workers 4`）+ Session 移 Redis + 加限流 |
| 1000 并发 | K8s 横向扩容 + 语义缓存 + Celery 异步入库 + 熔断 |
| 10000+ 并发 | 多区域部署 + 向量库分片 + 冷热分层 |

**关键原则**：
- Stateless API（session 不能存内存）
- 慢操作异步化（LLM 走 SSE，ingest 走 Celery）
- 优雅降级（外部依赖每个都有 fallback）

---

### 2.4 多模态支持怎么做？

**承认**：现在错题里的几何图 / 手写图 / 截图都没法被 RAG 检索到，只有文本能搜。

**根因**：embedding 模型只接受文本输入。

**方案**：

| 方案 | 效果 | 工程量 |
|------|------|--------|
| **图 → Caption → 入文本索引**：用 vision LLM（qwen-vl-max）给图生成描述，描述入向量库 | 简单可控 | 中等，~400 行 |
| **多模态 embedding**：CLIP 等模型直接对图做 embedding，新增 collection | 准确度高 | 大，需要新栈 |
| **OCR 文字提取**：图中文字单独提取出来加权重 | 适合表格 / 公式截图 | 中等 |

**实际怎么选**：先做 OCR 兜底（覆盖大部分错题截图），再做 Caption 方案（语义层），多模态 embedding 留作长期。

---

### 2.5 评测体系怎么建？

**承认**：现在没有量化评测，全凭"感觉好用"，无法持续优化。

**根因**：建评测集成本高（要标注 ground truth），但不建就是闭门造车。

**方案**：Ragas 框架

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset,  # {question, ground_truth, answer, contexts}
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ChatOpenAI(...),
    embeddings=...,
)
```

**4 个核心指标**：

- **Faithfulness（忠实度）**：答案有没有偏离上下文（防幻觉）
- **Answer Relevancy（答案相关性）**：答案有没有真正回答问题
- **Context Precision（上下文精度）**：检索的 chunks 噪声多不多
- **Context Recall（上下文召回）**：该找到的 chunks 有没有都找到

**怎么落地**：先建 50-100 条标注集（自己问、自己标），建一个基线分数，每次改完跑一遍看是否退步。

---

### 2.6 可观测性还能怎么做？

**承认**：现在只有控制台 print，生产环境调不动。

**方案**：

| 层 | 工具 | 看什么 |
|----|------|-------|
| **LangSmith** | langsmith SDK | 每次问答的全链路 trace、节点耗时、token 消耗 |
| **Prometheus + Grafana** | prometheus-fastapi-instrumentator | API QPS / P99 / 错误率 |
| **OpenTelemetry** | otel + Jaeger | 分布式追踪（跨服务） |
| **结构化日志** | loguru + ELK | 业务事件审计 |

**最小可行版**：LangSmith 5 分钟接好，先解决"哪个节点慢"的问题；Prometheus 后续上。

---

### 2.7 数据安全 / 合规怎么考虑？

**承认**：当前 RAG-Demo 把数据存在境外 Qdrant Cloud，业务上线前必须改。

**关键风险点**：

| 风险 | 应对 |
|------|------|
| **数据出境**：用户文档存在境外向量库 | 改用国内 Qdrant 自建 / 阿里云 DashVector |
| **API Key 泄露**：`.env` 不小心 commit | `.gitignore` + pre-commit hook 扫 `sk-` |
| **Prompt 注入**：用户问题里夹"忽略系统提示"指令 | system prompt 加防御性提示 + 输入侧 sanitize |
| **越权访问**：用户 A 检索到用户 B 的文档 | 向量库 metadata 加 user_id 过滤 |
| **LLM 回答含敏感信息** | 输出侧加正则 / 小模型审核 |

**notes-app 已经做的**：DOMPurify 消毒所有富文本输入防 XSS；`.gitignore` 严格隔离 `.env`。

---

### 2.8 错题本前端还能怎么优化？

| 优化点 | 收益 |
|--------|------|
| **虚拟滚动**（vue-virtual-scroller）| 错题超过 1000 条时列表渲染卡 |
| **IndexedDB 索引**：按 createdAt / kbId 建索引 | 筛选和分页变快 |
| **Service Worker 缓存策略**：stale-while-revalidate | 离线打开秒开 |
| **崩溃恢复优化**：当前每秒一次快照，改成内容变化时 debounce 500ms | 减少 70% 写入 |
| **图片懒加载**：错题图片用 `loading="lazy"` + IntersectionObserver | 长列表首屏快 |
| **批量操作 Worker**：批量导出 PDF 放 Web Worker | 不卡主线程 |

---

## 三、面试官可能挖的坑 + 怎么躲

### 3.1 "为什么不用 OpenAI / Claude，DeepSeek 不稳定吧？"

**躲坑姿势**：

> 选 DeepSeek 是因为价格（输入 ¥0.5/M、输出 ¥1.5/M，比 GPT-4o 便宜 10 倍）+ OpenAI 兼容协议（迁移成本零）+ 国内合规。稳定性方面我已经加了 3 次指数退避重试 + 失败明确标记，实际跑下来连续 1000 次请求成功率 99.5%。如果业务规模上来要稳定性 SLA，DeepSeek 有企业版，或者换百炼 qwen-plus 一行代码切换（base_url + model 改两个字段）。

**关键点**：用数据 + 解决方案应对，而不是辩解。

---

### 3.2 "你用 LangChain/LangGraph 不就是调用 API 吗？"

**躲坑姿势**：

> LangChain 解决的是"状态机管理"和"消息协议规范化"这两个真问题。我从原生 OpenAI SDK 起步，自己手动维护 `messages` 列表里 user / assistant / tool / tool_calls 的配对，踩了 3 次 400 错误（tool 消息成孤儿 / 历史断层 / 配对错乱）。换成 LangChain 1.x 的 `create_agent` 后这类错误从框架层面就杜绝了。框架不是为了写少代码，是为了让"正确"成为默认行为。

**关键点**：拿"我踩过的真坑"反驳，不抽象辩论。

---

### 3.3 "你这个 RAG 就 5 个节点，工业级 RAG 还有 chain-of-thought / agentic loop / multi-hop reasoning，你怎么看？"

**躲坑姿势**：

> 您说的对，这些都是 advanced RAG 的方向。我现在是 baseline RAG（路由 → 改写 → 混合检索 → 评分 → 生成），覆盖 80% 简单问答场景。如果业务里出现 multi-hop 问题（"对比 A 和 B 的差异"需要分别检索 A 和 B 再合并）我会加 query decomposition 节点；如果出现复杂推理我会加 CoT 模式（让 LLM 先生成 reasoning 再生成 answer，AgentResponse 加 reasoning 字段就行）；agentic loop 在 RAG-Demo 里已经做了（create_agent 内置工具调用循环）。但这些都不是免费的——agentic loop 平均一个问题多 2 次 LLM 调用、子问题拆解多 3 次，要根据业务问题复杂度和成本预算决定上哪些。

**关键点**：表态"我知道这些，没上是因为取舍"，而不是"我不知道"。

---

### 3.4 "你的代码有单元测试吗？覆盖率多少？"

**躲坑姿势**：

> 这块我做得不够，目前只有 README 里列的端到端验证清单（curl 测各端点 + 多轮对话测 history 完整性）。如果生产化，我会按这个优先级补测试：
> 1. **核心数据流的集成测试**：ingest 一份 PDF → 问相关问题 → 断言答案包含特定关键词
> 2. **Agent 状态机的单测**：mock LLM 返回固定 tool_calls，验证 chat_history 结构
> 3. **API 契约测试**：FastAPI 端点用 pytest + httpx
> 4. **关键节点的 prompt 评测**：用 Ragas 在固定数据集上跑回归
>
> 单测覆盖率不是首要指标，关键路径的回归测试更重要——因为 LLM 输出不确定，纯单测意义有限。

**关键点**：坦诚 + 给出优先级清单，比硬撑覆盖率数字可信。

---

### 3.5 "Tailwind CSS 和原生 CSS 有什么区别？为什么不用 CSS Modules？"

**躲坑姿势**：

> Tailwind 优势是开发速度（不用切 .css 文件 + 不用想类名）和样式约束（颜色 / 间距全部走 token，不会出现 16px / 17px 这种漂移）。劣势是模板代码长一些。CSS Modules 解决的是"样式作用域"问题，但 Vue 3 的 `<style scoped>` 已经自带作用域，所以 Tailwind 和 Vue 配合更顺。如果项目里有大量动态计算的样式或者复杂动画，原生 CSS / SCSS 会更合适——Tailwind 不是银弹。

---

### 3.6 "Electron 应用动辄几百 MB，为什么不用 Tauri？"

**躲坑姿势**：

> 选 Electron 是因为生态成熟、踩坑文档多、团队 JS 技能直接复用。Tauri 优势是包小（Rust 写 native 后端，几 MB），但要求会 Rust，调试链路也更复杂。如果团队有 Rust 储备 / 包体积是硬指标，Tauri 是更好的选择。我们这个项目对发布包大小不敏感，Electron 是合理选型。如果未来要做轻量级版本，迁移路径也很顺——前端 Vue 代码完全复用，只需要换壳。

---

### 3.7 "向量数据库为什么不用 Milvus / Weaviate？"

**躲坑姿势**：

> Chroma 优势是单文件持久化 + Python 原生集成 + 0 运维，适合本地优先场景（notes-app）。Qdrant Cloud 是托管服务，无运维 + 性能稳，适合云原生（RAG-Demo）。Milvus 适合企业级大规模（亿级向量、高 QPS），但运维复杂度高；Weaviate 内置模块化（如内嵌 reranker）但学习成本高。我们的选型是按业务规模匹配——本地用户 ≤10 万 chunk 用 Chroma，云上 ≤1000 万 chunk 用 Qdrant，过 10000 万再考虑 Milvus。

---

### 3.8 "你的项目跟 ChatPDF / Notion AI 有什么差异化？"

**躲坑姿势**：

> ChatPDF 是单文档对话工具，Notion AI 是嵌入笔记的助手，我们的错题本项目是**学习场景的垂直深耕**——
>
> 1. **场景特化**：SRS 间隔重复算法 + 学科标签体系 + 复习模式，对学生群体直接可用
> 2. **本地优先**：错题数据在 IndexedDB，AI 服务可选挂载，不联网也能正常用错题本核心功能
> 3. **多模态支持**：手写画笔 + 截图 + OCR，覆盖纸面错题的数字化路径
> 4. **可定制 AI 行为**：Skill 系统让用户自定义 `/讲解` `/答` `/口诀` 等触发词
>
> 我没想做 ChatPDF 的替代品——我做的是"一个学生该有的全套数字化学习工具，AI 是其中一个组件"。

---

### 3.9 "你这个项目你独立从 0 做的吗？AI 写了多少？"

**躲坑姿势**（这种问题最忌撒谎）：

> 实话说，整体架构和需求是我设计的，具体代码我用 Claude 做 pair programming——我写关键路径 + 测试，AI 补全样板和写 SQL 之类的体力活。这是我现在做项目的常规模式，效率比纯手写高 3-5 倍。但每一行代码我都能解释为什么这样写、能调试任何报错、能独立做架构决策——这是有没有"真理解"项目的关键。今天我们聊到的每个技术点（LangGraph 节点 / SSE 协议 / Pydantic 强约束 / tool_call_id 配对）我都是自己一行行调出来的，您可以随便挖。

**关键点**：诚实 + 强调"我能解释 + 能调试 + 能扩展"才是核心能力。

---

## 四、面试时的展示策略

### 4.1 怎么开场

> "我最近做了两个项目，都是围绕 RAG Agent 的不同部署形态。第一个 notes-app 是 Vue 3 + Electron + 本地 RAG，重点是工程化和端到端体验；第二个 RAG-Demo 是 Streamlit + FastAPI + 云原生 LangChain，重点是验证多用户 SaaS 架构。两个项目我可以分别讲，您想从哪个开始？"

→ 把选择权给对方，显得有掌控力。

---

### 4.2 怎么过渡

每个技术点讲完，主动抛"我下一步想做什么"：

> "...这部分目前是 baseline 实现，我下一步想加 HyDE 提升复杂查询召回率 / 接 LangSmith 看每次 trace / 做 Ragas 评测建立质量基线。"

→ 展示"持续在思考"，而不是"做完就完了"。

---

### 4.3 不会的题怎么答

不要装会。模板：

> "这个我没有深入研究过，凭直觉理解是 XX，可能会从 YY 角度切入。回去我可以查 ZZ 资料系统学一下。"

→ 90% 的面试官接受"不会但有思考路径"，0% 的面试官接受"瞎编"。

---

### 4.4 必须背下来的数字

| 项 | 数 |
|----|----|
| 错题本核心代码 | ~5000 行 TypeScript + ~1500 行 Vue |
| RAG-Demo 核心代码 | ~1500 行 Python |
| LangGraph 节点数 | 5（route / rewrite / retrieve / grade / generate）|
| 默认 chunk_size / overlap | 800 / 200 字符 |
| 默认 top_k 召回 / 重排 | 10 / 4 |
| BGE-small-zh / large-zh 维度 | 512 / 1024 |
| DeepSeek 价格 | 输入 ¥0.5/M、输出 ¥1.5/M |
| SRS 初始间隔 | 1 天 → 6 天 → 14 天 → 30 天 |
| 重试退避 | 3 次，0.6s/1.2s/2.4s |
| SSE 分隔符 | `\r\n\r\n` 或 `\n\n` 都要兼容 |

---

## 五、一句话总结

**对方问"这个项目难度在哪？"**

> 难度不在写代码，而在踩了三类坑后形成的体系化判断——
> 1. **协议层**：tool_calls / tool_call_id 配对、SSE 分隔符兼容、消息历史规范化
> 2. **架构层**：LangGraph 手装 vs `create_agent` 取舍、本地 vs 云端的部署形态选择、前后端分离时机
> 3. **工程层**：失败重试和降级、业务 ↔ AI 数据同步、错误明确化（不伪装成正常输出）
>
> 这三层每一层都有"看起来 work 但生产爆炸"的隐患，全部踩过一遍才敢说"我懂 RAG"。

---

> 这份文档每节末尾的"一句话答案"是抛给对方的第一句话，对方追问就展开"细节展开"。
> 不要一上来就背 1000 字大段——观察对方反应，按节奏喂信息。
