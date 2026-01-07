# 🚀 Advanced Context Manager — Zero-Loss Coverage-First v2.4.6

> Open WebUI Pipeline / 插件：长上下文管理 + 多模态处理 +（可选）自动记忆  
> 目标：在 token 预算内 **尽可能覆盖历史上下文**，同时保证稳定性、可观测性与可调优。

- **Version**: v2.4.6  
- **License**: MIT  
- **Repo**: https://github.com/JiangNanGenius

---

## 目录 / Table of Contents

- [中文文档](#cn)
  - [1. 核心亮点](#cn-1)
  - [2. 适用场景](#cn-2)
  - [3. 工作原理（高层）](#cn-3)
  - [4. 安装与启用](#cn-4)
  - [5. 快速开始（推荐参数）](#cn-5)
  - [6. Valves 配置说明](#cn-6)
  - [7. 可观测性（进度/统计/日志）](#cn-7)
  - [8. 性能与调优建议](#cn-8)
  - [9. 常见问题](#cn-9)
  - [10. 迁移指南](#cn-10)
  - [11. 更新日志](#cn-11)
- [English Documentation](#en)
  - [1. Highlights](#en-1)
  - [2. Use Cases](#en-2)
  - [3. How It Works (High-level)](#en-3)
  - [4. Install & Enable](#en-4)
  - [5. Quick Start (Recommended)](#en-5)
  - [6. Valves Reference](#en-6)
  - [7. Observability (Progress/Stats/Logs)](#en-7)
  - [8. Performance & Tuning](#en-8)
  - [9. FAQ](#en-9)
  - [10. Migration](#en-10)
  - [11. Changelog](#en-11)
  - [License / Credits](#en-license)

---

<a id="cn"></a>

# 中文文档

<a id="cn-1"></a>

## 1. 核心亮点

**Coverage-First（覆盖优先）** + **Zero-Loss（零丢失兜底）**：

- **覆盖优先**：优先保证历史内容“被覆盖到”（原文/摘要/简化摘要至少一种）。
- **自适应分块**：按 token、角色切换、时间间隔、内容类型（代码/文本）等切块，避免碎片化。
- **一次性比例缩放**：对摘要长度做统一缩放，减少多轮抖动与超预算风险。
- **升级池（Upgrade Pool）**：预留预算把关键内容升级为原文，确保关键事实可追溯。
- **双重护栏**：
  - 护栏 A：映射校验 + 范围合并，保证“应该落地的”都落地；
  - 护栏 B：未落地部分生成“简化摘要”兜底，最大化覆盖率（可计算）。
- **Top-up 填窗**：在满足护栏后，贪心填充更多原文/重要块，把窗口利用率拉到目标区间。
- **多模态支持**：可选择直传多模态、视觉转文本、向量检索（RAG）。
- **高可观测性**：阶段进度、统计指标、缓存命中、并发数、摘要/向量请求数、覆盖率等。

> v2.4.6 侧重兼容性与健壮性：对 Memory 404、LLM 非 JSON 输出、OpenAI-compatible 兼容路径等更稳。

---

<a id="cn-2"></a>

## 2. 适用场景

- 会话非常长，需要“尽量不丢历史信息”，尤其是**需求/约束/参数**反复引用的场景。
- RAG 召回不稳定，想要“召回 + 覆盖”混合策略。
- 需要对上下文构建过程有更强可控性（预算、并发、摘要强度、是否保原文等）。
- 多模态（含图片）对话，希望在“可用 token”内更可靠地融合。

---

<a id="cn-3"></a>

## 3. 工作原理（高层）

一个典型的组装流程（示意）：

```text
历史消息
  → 排序/打标（稳定 ID）
  →（可选）多模态预处理（图片→描述/标签）
  → 轻筛（时间/角色/权重/规则）
  →（可选）向量召回（两阶段：粗召回 → 精召回）
  → Coverage 规划（micro 摘要 + block 摘要）
  → 统一缩放（按预算缩放摘要长度）
  → 升级池（关键消息升级原文）
  → 并发生成摘要（带缓存）
  → 双重护栏组装（确保覆盖）
  → Top-up 填窗（尽量用满预算）
  → 输出给模型
```

你可以把它理解成：  
**先保证“不丢信息” → 再保证“可读/成本可控” → 最后“把窗口用满”。**

---

<a id="cn-4"></a>

## 4. 安装与启用

以 Open WebUI Pipelines 为例：

1. **Settings → Pipelines**
2. **New Pipeline (+)** / 或导入 pipeline 文件
3. 粘贴 v2.4.6 源码并保存
4. 在会话中启用该 Pipeline（置顶或按会话选择）

> 如果你的环境有多套 OpenAI-compatible API（如代理、不同模型供应商），建议先确认基础对话可用，再开启向量与记忆功能。

---

<a id="cn-5"></a>

## 5. 快速开始（推荐参数）

下面是“推荐思路”的默认参数组合（实际字段名以你的代码中 `Valves` 为准）：

- **目标窗口利用率**：`0.80 ~ 0.88`（建议 0.85）
- **并发摘要**：`4 ~ 8`
- **摘要强度**：中等（先覆盖，再压缩）
- **升级池比例**：`0.10 ~ 0.20`（关键内容保原文）
- **Top-up**：开启
- **护栏 B**：开启（零丢失兜底）

---

<a id="cn-6"></a>

## 6. Valves 配置说明

> ⚠️ 注意：不同版本/分支字段名可能略有差异。以下按“常见实现”整理，你可以直接在代码里搜索 `class Valves` 或 `VALVES` 对照调整。

### 6.1 预算 / Token 相关

- `max_context_tokens`：最大可用上下文 token（或由模型识别后自动推导）
- `target_utilization`：目标利用率（例如 0.85）
- `safety_margin`：安全余量（防止估算偏差）

### 6.2 覆盖策略 / 摘要策略

- `coverage_mode`：覆盖模式（例如 `coverage_first`）
- `micro_summary_tokens`：单条 micro 摘要 token 上限
- `block_summary_tokens`：block 摘要 token 上限
- `one_shot_scaling`：是否统一缩放摘要长度（建议开启）
- `guardrail_b_enabled`：是否启用简化摘要兜底（建议开启）

### 6.3 升级池（保原文）

- `upgrade_pool_ratio`：预留预算比例（建议 0.10~0.20）
- `upgrade_priority_rules`：升级规则（例如：系统/开发者消息优先、含关键指令/参数的消息优先）

### 6.4 Top-up 填窗

- `topup_enabled`：是否启用填窗（建议开启）
- `topup_strategy`：填窗策略（优先升级 micro → 再贪心加入未落地原文/块）

### 6.5 RAG / 向量召回（可选）

- `rag_enabled`：是否启用向量召回
- `rag_k`：召回条数
- `rag_two_stage`：两阶段召回（粗筛→精排）
- `embedding_model`：embedding 模型名
- `embedding_cache_ttl`：embedding 缓存 TTL

### 6.6 Memory / 自动记忆（可选）

- `memory_enabled`：是否启用记忆
- `memory_write_mode`：写入策略（只写高置信/只写明确偏好/全量等）
- `memory_404_ok`：记忆为空（404）是否视为正常（v2.4.6 建议为 true）

### 6.7 多模态（可选）

- `multimodal_mode`：`pass_through` / `vision_to_text` / `multimodal_rag`
- `image_preprocess`：图片预处理（缩放、去噪、描述生成等）

---

<a id="cn-7"></a>

## 7. 可观测性（进度/统计/日志）

常见输出（示例）：

- **阶段进度**：`stage=chunking / retrieval / summarizing / assembling / topup`
- **覆盖率**：覆盖了多少历史消息（原文/摘要/简化摘要）
- **窗口利用率**：最终上下文 token 使用率
- **缓存命中**：摘要缓存、embedding 缓存命中
- **并发情况**：摘要并发数、排队/超时情况
- **退化策略触发**：是否进入“全局块摘要/强压缩”等兜底模式

建议：
- 开启 debug 时，先观察 5~10 次真实对话的统计，再决定调参方向。
- 若你经常看到“超预算 + 截断”，应优先降低 `target_utilization` 或增加 `safety_margin`。

---

<a id="cn-8"></a>

## 8. 性能与调优建议

### 8.1 先解决“不稳”，再追求“更省钱”

- **优先**：打开护栏、打开统一缩放、合理 safety margin
- **其次**：再调小摘要 token、提高缓存 TTL、减少 RAG 的 k

### 8.2 常见调参路径

- **输出太长/超预算**：降低 `target_utilization`（0.85 → 0.80），或提高 `safety_margin`
- **关键内容经常“被摘要掉”**：提高 `upgrade_pool_ratio`，并增加升级规则
- **摘要成本太高**：减少 `block_summary_tokens`，降低并发，增加缓存 TTL
- **RAG 噪声多**：降低 `rag_k`，开启 two-stage，增加过滤规则（时间/角色/主题）

---

<a id="cn-9"></a>

## 9. 常见问题

### Q1：为什么我觉得“仍然丢信息”？
- 先确认是否开启 **护栏 B**（简化摘要兜底）。
- 检查是否启用了“保险截断/硬截断”。如果你更想要“宁可多摘要也不截断”，应把截断策略调成更保守。

### Q2：为什么日志里会出现 Memory 404？
- v2.4.6 起通常会把“没有记忆”视为正常，不应影响主流程。若你看到报错，请检查 `memory_404_ok` 类似字段是否开启。

### Q3：LLM 输出不是 JSON 导致解析失败怎么办？
- v2.4.6 通常已增强兼容：code fence、单引号、尾随逗号等。如果仍失败，建议在摘要提示词中强制：
  - “只输出 JSON，不要 Markdown”
  - “字段必须存在，即使为空也要给空数组/空字符串”

### Q4：并发高时偶发超时？
- 降低并发 `summary_concurrency`，或者提高超时阈值；
- 开启缓存，减少重复摘要请求；
- 若使用代理，检查代理的连接复用与限流策略。

---

<a id="cn-10"></a>

## 10. 迁移指南

从 v2.4.5 → v2.4.6：

- **Memory**：建议把“记忆为空的 404”视为正常（避免误报）。
- **OpenAI-compatible**：更推荐走统一 `chat.completions.create` 路径，不依赖 `.parse()`。
- **Memory 更新字段**：兼容 `content / new_content` 的别名，旧数据无需重写。

---

<a id="cn-11"></a>

## 11. 更新日志

### v2.4.6
- Memory：对空记忆导致的 404 做兼容处理，避免误报
- LLM 输出解析：兼容更多非标准 JSON 形式（如 code fence/单引号/松散格式）
- OpenAI-compatible：统一兼容调用路径（避免依赖 `.parse()`）
- Memory Update：兼容 `content / new_content` 字段别名
- 执行链路：增加 DB fallback 与更清晰日志（便于排查）

---

---

<a id="en"></a>

# English Documentation

<a id="en-1"></a>

## 1. Highlights

**Coverage-First** + **Zero-Loss fallback**:

- **Coverage-First**: prioritize covering as much historical context as possible before aggressive compression.
- **Adaptive chunking**: chunk by token size, role shifts, time gaps, and content types (code vs text).
- **One-shot scaling**: scale micro/block summaries once to fit budget—reduces oscillation and overshoot.
- **Upgrade pool**: reserve budget to promote critical messages back to raw text.
- **Dual guardrails**:
  - Guardrail A: mapping checks + range merge, ensuring intended content is included;
  - Guardrail B: generate simplified fallback summaries for uncovered segments.
- **Top-up filler**: after guardrails, greedily fill remaining budget to reach target utilization.
- **Multimodal**: pass-through, vision-to-text, or multimodal RAG.
- **Observability**: progress stages, coverage ratio, utilization, cache hit, concurrency, request counts.

---

<a id="en-2"></a>

## 2. Use Cases

- Very long chats where you need **maximum retention** of constraints, specs, parameters, and decisions.
- Hybrid strategy: recall via RAG + guaranteed coverage of critical history.
- You want explicit control over budget, concurrency, summary strength, and raw-text preservation.
- Multimodal conversations (images) that must remain useful under tight context budgets.

---

<a id="en-3"></a>

## 3. How It Works (High-level)

```text
History
  → stable ordering / IDs
  → (optional) multimodal preprocessing
  → lightweight filtering
  → (optional) vector retrieval (two-stage)
  → coverage planning (micro + block)
  → one-shot scaling
  → upgrade pool (promote raw text)
  → concurrent summarization (cached)
  → dual-guardrail assembly
  → top-up filling
  → output
```

---

<a id="en-4"></a>

## 4. Install & Enable

In Open WebUI:

1. Settings → Pipelines
2. New Pipeline (+) / import a pipeline file
3. Paste v2.4.6 source code and save
4. Enable it per chat (or pin it globally)

---

<a id="en-5"></a>

## 5. Quick Start (Recommended)

Suggested baseline (adjust to your code’s `Valves`):

- Target utilization: `0.80 ~ 0.88` (recommended 0.85)
- Summary concurrency: `4 ~ 8`
- Upgrade pool ratio: `0.10 ~ 0.20`
- Guardrail B: ON
- Top-up: ON
- One-shot scaling: ON

---

<a id="en-6"></a>

## 6. Valves Reference

> Field names may differ across forks/versions. Search for `Valves`/`VALVES` in code.

### Budget / Tokens
- `max_context_tokens`
- `target_utilization`
- `safety_margin`

### Coverage / Summaries
- `coverage_mode`
- `micro_summary_tokens`
- `block_summary_tokens`
- `one_shot_scaling`
- `guardrail_b_enabled`

### Upgrade Pool
- `upgrade_pool_ratio`
- `upgrade_priority_rules`

### Top-up
- `topup_enabled`
- `topup_strategy`

### RAG (Optional)
- `rag_enabled`
- `rag_k`
- `rag_two_stage`
- `embedding_model`
- `embedding_cache_ttl`

### Memory (Optional)
- `memory_enabled`
- `memory_write_mode`
- `memory_404_ok` (recommended true in v2.4.6)

### Multimodal (Optional)
- `multimodal_mode`
- `image_preprocess`

---

<a id="en-7"></a>

## 7. Observability (Progress/Stats/Logs)

Typical metrics:

- progress stage: `chunking / retrieval / summarizing / assembling / topup`
- coverage ratio
- utilization ratio
- cache hits (summary/embedding)
- concurrency level
- fallback/degradation triggers

---

<a id="en-8"></a>

## 8. Performance & Tuning

- Fix stability first: guardrails + one-shot scaling + sufficient safety margin.
- If overshooting budget: lower `target_utilization`, increase `safety_margin`.
- If critical info gets summarized too often: increase `upgrade_pool_ratio`, refine upgrade rules.
- If summary cost is high: reduce summary token caps, increase cache TTL, reduce concurrency.
- If RAG is noisy: lower `rag_k`, enable two-stage, add filters.

---

<a id="en-9"></a>

## 9. FAQ

**Q: Still losing information?**  
A: Ensure Guardrail B is enabled and truncation is conservative.

**Q: Memory 404?**  
A: v2.4.6 typically treats “no memory” as normal; check `memory_404_ok`.

**Q: JSON parse failures?**  
A: Strengthen prompts to force pure JSON output; v2.4.6 already improves tolerance.

---

<a id="en-10"></a>

## 10. Migration

From v2.4.5 → v2.4.6:
- treat memory-empty 404 as OK
- prefer unified OpenAI-compatible `chat.completions.create`
- accept `content / new_content` aliases for memory updates

---

<a id="en-11"></a>

## 11. Changelog

### v2.4.6
- Memory: treat empty-memory 404 as normal
- Robust parsing for non-strict JSON from LLM
- Unified OpenAI-compatible call path
- Memory update field aliasing: `content / new_content`
- clearer execution logs + DB fallback

---

<a id="en-license"></a>

## License / Credits

MIT License.  
Credits to the project author(s) and contributors.

