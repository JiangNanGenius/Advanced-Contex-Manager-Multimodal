# 🚀 高级上下文管理器（多模态 + 上下文最大化）v2.6.3  
Advanced Context Manager (Multimodal + Context Window Maximization) v2.6.3

**作者 / Author**: JiangNanGenius  
**版本 / Version**: 2.6.3  
**License**: MIT  
**Open WebUI 最低版本 / Required Open WebUI Version**: 0.5.17  
**GitHub**: https://github.com/JiangNanGenius  

---

## 中文说明（ZH）

### 1) 这是什么？
这是一个用于 **Open WebUI Filter** 的高级上下文管理器，目标是：  
- 在**上下文窗口有限**时，尽量“多保留、多覆盖、少丢失”历史内容  
- 支持**多模态**（图片）预处理：图片转写/描述 → 变成可检索可总结的文本  
- 内置 **Auto Memory 自动记忆**：后台运行、可不在前台显示状态

适用于：长对话、复杂技术排障、代码/配置密集型对话、多轮上下文依赖强的场景。

---

### 2) 核心能力
#### ✅ 上下文最大化（Coverage-First）
- **Coverage 分档**：高/中/低权重消息  
- **微摘要 + 块摘要**：按预算为每条消息/消息块生成摘要，尽量覆盖全部要点  
- **自适应分块**：根据原文 token、内容连续性、角色切换、分数波动动态分块  
- **升级策略**：用“升级池”把高价值摘要恢复成原文（更接近“不截断”）  
- **不截断保障（Zero-Loss Guarantee）**：通过预算调整/兜底策略减少硬截断风险

#### ✅ 多模态（图片）处理
- 检测消息中图片（支持 URL / data:base64）  
- 可选：在多模态模型支持时 **保留原图**，否则先做 **Vision 转写** 再进入摘要/检索  
- 图片 URL 有严格校验与清洗，避免异常输入导致流程崩溃

#### ✅ 记忆系统（Auto Memory，后台）
- 自动检索相关记忆 → 让 LLM 决定 add/update/delete  
- 强制写入前缀：`记住:` / `remember:`（命中则跳过 LLM 判断，直接写入）  
- 输出严格 JSON Schema，包含 actions 与 reason，便于排障  
- 兼容 Open WebUI 常见行为：**用户无记忆时 query 可能 404**（视为正常）

#### ✅ 性能与稳定性
- EmbeddingCache：向量缓存（按 content_key 复用）  
- 并发控制：Semaphore 限制最大并发请求  
- 安全 API 调用：失败重试、超时、降级兜底  
- MessageOrder：稳定消息 ID、稳定顺序，降低“乱序/映射丢失”风险  
- 统计信息：处理耗时、命中率、摘要次数、fallback 次数、覆盖率等

---

### 3) 运行依赖
- Open WebUI：>= **0.5.17**
- Python 依赖（按需）：
  - `openai`（AsyncOpenAI）✅（你代码里通过 OPENAI_AVAILABLE 控制）
  - `httpx`（可选）
  - `tiktoken`（可选，用于更准 token 估算；没有则退化为字符估算）
  
> 如果日志提示 `OPENAI_AVAILABLE=False`，说明没有安装 openai 包或导入失败。

---

### 4) 安装方式（常见做法）
> 不同 Open WebUI 部署方式路径略有差异，下面给最常见的两种。

#### A. 通过 Open WebUI 后台（如果你的版本支持 Filter/Plugin 粘贴）
1. 进入 Admin / 管理后台  
2. 找到 Filters / 自定义过滤器（或类似入口）  
3. 新建 / 上传该脚本  
4. 保存并重启相关服务（如需要）

#### B. Docker / 本地挂载（更通用）
1. 将脚本保存为一个 `.py` 文件（例如：`advanced_context_manager_v2_6_2.py`）  
2. 放到 Open WebUI 后端可加载 Filters 的目录（示例：`/app/backend/open_webui/filters/` 或你的自定义 filters 目录）  
3. 重启容器 / 服务

> 如果你告诉我你是 Docker 版还是源码版，以及容器内 Open WebUI 后端目录结构，我可以把挂载路径写成“完全可复制”的命令。

---

### 5) 快速配置（Valves 重点项）
下面是你最常需要改的几类配置（都在 `Filter.Valves`）：

#### 5.1 API 与模型
- `api_base`：OpenAI-compatible Base URL（例如火山/代理/自建网关）
- `api_key`：密钥
- `text_model`：文本摘要/检索相关调用
- `multimodal_model`：图片转写/多模态摘要
- `memory_model`：记忆决策模型
- `text_vector_model` / `multimodal_vector_model`：向量模型

#### 5.2 Token 与预算策略
- `default_token_limit` / `max_fallback_token_limit` / `token_safety_ratio` / `target_window_usage`
- `response_buffer_ratio` / `response_buffer_min/max`
- `max_window_utilization` / `min_preserve_ratio`
- `enable_zero_loss_guarantee` / `max_budget_adjustment_rounds`

#### 5.3 Coverage 相关
- `coverage_high_score_threshold` / `coverage_mid_score_threshold`
- `coverage_high_summary_tokens` / `coverage_mid_summary_tokens`
- `coverage_block_summary_tokens`
- `raw_block_target` / `max_blocks` / `upgrade_min_pct`

#### 5.4 多模态策略
- `enable_multimodal`
- `preserve_images_in_multimodal`
- `always_process_images_before_summary`
- `vision_prompt_template` / `vision_max_tokens`

#### 5.5 Auto Memory
- `enable_auto_memory`
- `memory_messages_to_consider`
- `memory_related_memories_n`
- `memory_force_add_prefixes`（默认：`记住:;remember:`）
- `override_memory_context`

#### 5.6 模型能力识别与兜底（当前实现）
- 先走规则识别（`ModelMatcher.match_model`，正则匹配常见家族）
- 识别失败时：使用默认能力参数（`200k` 上下文、文本模型、默认图片 token 预算）并输出提示
- 运行时错误学习：从 API 报错中提取能力信号（`limit` / `multimodal` / `image_tokens`）
  - 先做正则抽取（中英文错误）
  - 再用文本模型做结构化解析（JSON）
- 学到的能力会覆盖本次会话内对应模型的初始识别结果（`runtime override`）

> 说明：当前版本已移除“大型静态精确模型字典”，改为“规则识别 + 失败默认 + 错误学习覆盖”。

---

### 6) 使用说明
- 正常情况下无需手动触发：当检测到  
  - token 超限（历史消息太长），或  
  - 当前消息包含图片但模型不支持多模态  
  就会进入处理流程：分片 → 检索/评分 → Coverage 计划 → 摘要生成 → 输出组装。

- Auto Memory 默认后台运行：  
  - 用户最新消息命中 `记住:` 前缀时会立刻写入记忆  
  - 否则会检索相关记忆 → 让 LLM 决定是否 add/update/delete

---

### 7) 排障建议（高频问题）
1) **“No Function class found in the module”**  
- 确认文件里顶层类名为 `Filter`（你这里是 `class Filter:` ✅）  
- 确认 Open WebUI 对 Filter 的加载规则：有些版本要求固定导出结构/目录位置  
- 确认脚本无语法错误（尤其是复制粘贴截断）

2) **记忆查询 404**  
- 代码已按“用户无记忆时返回 404”为正常处理（会日志提示但不中断）

3) **LLM 返回非 JSON 导致记忆解析失败**  
- 你代码里已经做了 code fence 清理、JSON 截取、以及 “no action” 文本降级  
- 若仍失败：把 `debug_level` 提升到 2 或 3，查看 raw preview

4) **处理太慢 / API 调用太多**  
- 降低 `vector_top_k`、`rerank_top_k`  
- 降低 `max_concurrent_requests`（避免把网关打爆）  
- 调高相似度阈值：`text_similarity_threshold` / `multimodal_similarity_threshold`

---

### 8) License
MIT License. 详见项目 License 文件或仓库说明。

---

## English (EN)

### 1) What is this?
This is an **Open WebUI Filter** that maximizes useful context under limited context windows:
- Preserve and cover as much conversation history as possible with **Coverage-First planning**
- Support **multimodal (image) preprocessing** by transcribing/describing images into searchable text
- Run **Auto Memory** in the background (optionally silent in the frontend)

Best for long technical chats, code/config heavy sessions, and multi-turn reasoning.

---

### 2) Key Features
#### ✅ Context Window Maximization (Coverage-First)
- Score history messages and classify into high/mid/low priority
- Generate **micro-summaries** (per message) and **block summaries** (per adaptive block)
- Adaptive blocking by token size, continuity, role boundaries, and score changes
- Upgrade strategy: reserve an “upgrade pool” to restore high-value content back to raw text
- **Zero-Loss Guarantee** style budgeting to reduce hard truncation risk

#### ✅ Multimodal Support
- Detect images in message content (URL or base64 `data:`)
- Optionally keep original images for multimodal-capable models, otherwise do vision-to-text first
- Strict URL validation and sanitization for robustness

#### ✅ Auto Memory (Background)
- Retrieve related memories → ask the LLM to add/update/delete
- Forced prefix write: `记住:` / `remember:` (bypass LLM decision, directly add memory)
- Strict JSON schema output with `actions` + `reason` for debugging
- Compatible with Open WebUI behavior where querying memories may return **404 if none exist**

#### ✅ Performance & Stability
- EmbeddingCache for reusing embeddings
- Concurrency control via semaphore
- Safe API calls with retry/timeout and fallbacks
- Stable message ordering/IDs to avoid mapping loss
- Detailed processing stats (coverage, cache hits, requests, fallbacks, etc.)

---

### 3) Requirements
- Open WebUI: **>= 0.5.17**
- Optional Python packages:
  - `openai` (AsyncOpenAI)
  - `httpx`
  - `tiktoken` (better token estimation; falls back if missing)

---

### 4) Installation (Common Approaches)
#### A) Via Open WebUI Admin UI (if supported)
1. Open Admin panel  
2. Go to Filters / Custom Filters  
3. Create / upload this script  
4. Save and restart if needed

#### B) Docker / Local Mount
1. Save the script as a `.py` file (e.g. `advanced_context_manager_v2_6_2.py`)  
2. Put it into the backend filters directory used by your deployment  
3. Restart the service/container

> If you tell me your deployment type (Docker vs source) and backend directory layout, I can provide exact copy-paste mount commands.

---

### 5) Configuration (Valves Highlights)
- API & models: `api_base`, `api_key`, `text_model`, `multimodal_model`, `memory_model`, vector models
- Token budgeting: `default_token_limit`, `max_fallback_token_limit`, `token_safety_ratio`, `target_window_usage`, response buffer
- Coverage planning: thresholds, per-summary budgets, block sizing, upgrade pool
- Multimodal: preserve images vs vision preprocessing
- Auto Memory: messages to consider, related memories k, forced prefixes, override memory context

#### 5.1 Model capability handling (current)
- First-pass rule-based recognition (`ModelMatcher.match_model`) using regex family patterns
- On recognition miss: fallback to safe defaults (200k context, text-mode defaults) and emit a hint
- Runtime learning from API errors (`limit` / `multimodal` / `image_tokens`):
  - regex extraction (CN/EN error texts)
  - text-model JSON extraction
- Learned signals are applied as runtime overrides for the same model key in-session

> Note: the large static exact model dictionary has been removed in favor of
> “rule-based recognition + default fallback + error-driven runtime learning”.

---

### 6) How it works
The filter runs automatically when:
- conversation history exceeds the target token budget, or
- images appear but the selected model is not multimodal

Pipeline (simplified):
chunking → scoring/retrieval → coverage planning → summary generation → guarded assembly → output

Auto Memory runs in the background:
forced-prefix add OR (retrieve → LLM action plan → apply).

---

### 7) Troubleshooting
- “No Function class found…”: ensure top-level class is `Filter` and the file is fully copied (no truncation)
- Memory query 404: treated as normal when no memories exist
- Non-JSON LLM output: the code already strips fences and extracts JSON; increase `debug_level` for raw preview
- Too slow: reduce `vector_top_k` / rerank top-k, lower concurrency, increase similarity thresholds

---

## Changelog (简要)
- v2.6.3: 稳定消息 ID / 更强的覆盖摘要与预算策略 / Auto Memory 后台机制增强 / 缓存与并发稳定性提升  
- v2.6.x: 多模态预处理与兜底策略强化、统计与日志更完整

---

## Credits
JiangNanGenius and contributors.

---
