```markdown
# 🚀 Advanced Context Manager — Zero-Loss Coverage-First v2.4.6

> Open WebUI 插件 / Pipeline 组件 · 智能长上下文管理 + 多模态处理 · **零丢失保障（可配置）**

**Author:** JiangNanGenius  
**Version:** 2.4.6  
**License:** MIT  
**Repo:** https://github.com/JiangNanGenius

---

## 目录 | Table of Contents

* [中文文档](#中文文档)
  * [核心亮点](#核心亮点)
  * [架构总览](#架构总览)
  * [安装](#安装)
  * [快速开始](#快速开始)
  * [关键配置（Valves）](#关键配置valves)
  * [工作流程](#工作流程)
  * [事件--进度--统计](#事件--进度--统计)
  * [常见问题](#常见问题)
  * [性能与调优建议](#性能与调优建议)
  * [迁移指南（从旧版本）](#迁移指南从旧版本)
  * [更新日志](#更新日志)
* [English Documentation](#english-documentation)
  * [Key Highlights](#key-highlights)
  * [Architecture Overview](#architecture-overview)
  * [Installation](#installation)
  * [Quick Start](#quick-start)
  * [Essential Config (Valves)](#essential-config-valves)
  * [Pipeline Flow](#pipeline-flow)
  * [Events--Progress--Stats](#events--progress--stats)
  * [FAQ](#faq)
  * [Performance--Tuning](#performance--tuning)
  * [Migration Guide](#migration-guide)
  * [Changelog](#changelog)
  * [License--Credits](#license--credits)

---

# 中文文档

## 核心亮点

**v2.4.6** 在 **Coverage-First（覆盖优先）+ Zero-Loss（零丢失）** 的基础上，继续强化“可解释覆盖 / 可观测处理 / 兼容性与健壮性”：

* **Coverage-First 覆盖优先策略**：先覆盖后优化，确保历史内容尽可能完整地进入模型窗口。
* **自适应分块（Adaptive Blocks）**：按原文 Token 量 + 角色切换 + 时间间隔 + 分数突变智能切块，控制块数与成本。
* **一次性比例缩放**：根据总预算对 *micro 摘要* 与 *块摘要* 统一缩放，避免多轮抖动与超支。
* **升级池保护（Preserve Upgrade Pool）**：预留预算按“价值密度”将关键消息升级为原文，防止摘要过度。
* **双重护栏组装**：
  * 护栏 A：映射校验与范围合并，防止遗漏；
  * 护栏 B：未落地条目提供“简化摘要”兜底，保证覆盖率可计算、可证明。
* **Top-up 窗口填充**：先把已选 micro 升级为原文，再贪心加入未落地原文，将窗口利用率拉到目标（默认 85%）。
* **零丢失保障（可配置）**：可禁用“保险截断”，即使超限也尽量不截断；并通过护栏 B 提供兜底摘要。
* **进度与统计**：阶段化进度条、覆盖率/窗口使用率、缓存命中、并发数、摘要/向量请求数等全面指标。
* **多模态策略**：支持直传多模态 / 视觉转文本 / 多模态向量 RAG；图片先验清洗、描述、注入摘要流程。
* **模型智能识别**：广覆盖的 Model Matcher（GPT/Claude/Qwen/Doubao/GLM 等），自动 Token 限制与安全系数。
* **向量缓存与并发**：Embedding 缓存 + 并发拉取，配合轻量级初筛，两阶段召回更快更稳。
* **v2.4.6 兼容性增强（重点）**：
  * Memory 相关：对“用户暂无记忆导致的 404”视为正常，避免误报；
  * LLM 输出：兼容非 JSON / code fence / 单引号等常见返回，解析更稳；
  * OpenAI-compatible：统一走 `chat.completions.create`，避免依赖 `.parse()`；
  * Memory Update：兼容 `content / new_content` 字段别名；
  * 记忆执行：增加 DB fallback 路径与更直观日志，排障更容易。

> 与旧版（如 1.x / 2.4.1~2.4.5）相比，2.4.6 更强调 **健壮性**、**兼容性** 与 **可观测性**。

---

## 架构总览

```

原始消息 → MessageOrder 打标 → 多模态预处理 →
相关度两阶段召回（轻筛 → 向量）→ Coverage 规划（micro + block）→
比例缩放 → 升级池（原文升级）→ 并发生成摘要 → 双重护栏组装 →
Top-up 填窗 → 用户消息保护 → 出口

````

主要组件：

* **MessageOrder**：稳定 ID/顺序标记；支持分片 ID（`msg#k`）。
* **TokenCalculator**：文本/图片 Token 粗估；模型安全系数（默认 92%）。
* **MessageChunker**：段落/句子/代码友好的智能分片。
* **CoveragePlanner**：分档 + 自适应块 + 统一缩放 + 极端退化（全局块）。
* **ProgressTracker**：阶段进度、漂亮进度条、前端事件。
* **ProcessingStats**：全链路指标与覆盖率计算。
* **Top-up Filler**：窗口利用率拉升到目标值。

---

## 安装

> 以 **Open WebUI** 为例（Settings → Pipelines）：

1. 在 **Pipelines** 点击 **+** 新建管道。
2. 将 `v2.4.6` 源码粘贴到编辑器，保存。
3. 置顶到所需工作流，或为特定会话启用。

> 依赖：可选的 `tiktoken` / `httpx` / `openai` SDK（仅当你启用外部向量或摘要时）。未安装时将自动降级到本地估算与兜底。

---

## 快速开始

### 1) Pipeline Header（建议保留在文件顶部）

> 复制到你的 Pipeline 源码顶部（`title/author/version/...` 会被 Open WebUI 识别）

```python
"""
title: 🚀 Advanced Context Manager - Zero-Loss Coverage-First v2.4.6
author: JiangNanGenius
version: 2.4.6
license: MIT
required_open_webui_version: 0.5.17
Github: https://github.com/JiangNanGenius
description: 智能长上下文管理 + 多模态处理 + 可选自动记忆（后台运行，可配置前台状态显示）
"""
````

### 2) 最小配置（默认即可运行）

> 多数情况下不需要改任何东西。下面是常用核心项的最小示例（Valves 节选）：

```yaml
enable_processing: true
token_safety_ratio: 0.92
target_window_usage: 0.85

# 建议：一般保持开启
enable_coverage_first: true
enable_zero_loss_guarantee: true
disable_insurance_truncation: true

# 调试：需要排障时再开
debug_level: 0
```

### 3) 如需使用外部 API（可选）

```yaml
api_base: "https://ark.cn-beijing.volces.com/api/v3"
api_key:  "<YOUR_API_KEY>"

text_model: "doubao-1-5-lite-32k-250115"
multimodal_model: "doubao-1.5-vision-pro-250328"

text_vector_model: "doubao-embedding-large-text-250515"
multimodal_vector_model: "doubao-embedding-vision-250615"
```

> 未配置 API 时，依然可运行核心逻辑（轻量级召回、智能分块、Top-up、护栏组装等），但“高质量摘要/向量检索”会降级为本地启发式。

---

## 关键配置（Valves）

> 字段很多，下列为 **最常用** / **影响大** 的参数。其余请参考源码默认值（`Filter.Valves` 已注释）。

### 基础控制

* `enable_processing`: 是否启用整个处理链（默认 `true`）
* `excluded_models`: 排除不处理的模型（逗号分隔）
* `debug_level`: 0~3，越高日志越详细
* `suppress_frontend_when_idle`: 无需处理时不显示前端状态（默认 `true`）

### 窗口与预算

* `token_safety_ratio`: 模型限制的安全系数（默认 `0.92`）
* `target_window_usage`: 期望窗口使用率（默认 `0.85`）
* `response_buffer_ratio|min|max`: 预留回答空间（默认 6%，介于 1000~3000 tokens）

### Coverage-First

* `coverage_high_score_threshold`: 高权重阈值（默认 `0.7`）
* `coverage_mid_score_threshold`: 中权重阈值（默认 `0.4`）
* `coverage_high_summary_tokens`: 高权重 micro 预算（默认 `100`）
* `coverage_mid_summary_tokens`: 中权重 micro 预算（默认 `50`）
* `coverage_block_summary_tokens`: 块摘要目标预算（默认 `350`）
* `upgrade_min_pct`: 升级池最小预留比例（默认 `0.2`）

### 自适应分块

* `raw_block_target`: 单块原文目标 tokens（默认 `15000`）
* `max_blocks`: 最大块数（默认 `8`）
* `floor_block`: 块摘要最小预算（默认 `300`）

### 多模态

* `enable_multimodal`: 开关（默认 `true`）
* `preserve_images_in_multimodal`: 多模态模型是否保留原图（默认 `true`）
* `always_process_images_before_summary`: 摘要前先做视觉处理（默认 `true`）

### 保护与截断（零丢失相关）

* `force_preserve_current_user_message`: 强制保留当前用户消息（默认 `true`）
* `preserve_recent_exchanges`: 保护最近 N 轮对话（默认 `4`）
* `enable_zero_loss_guarantee`: 启用“尽量不截断”（默认 `true`）
* `disable_insurance_truncation`: 禁用保险截断（默认 `true`，更偏向零丢失）

---

## 工作流程

1. **打标与分离**：对消息打 `_order_id`，分离系统/历史/当前用户消息。
2. **多模态处理**：根据模型能力与预算选择直传 / 视觉转文本 / RAG。
3. **两阶段召回**：文字轻筛 → 小集合向量化 → 并发计算相似度。
4. **Coverage 规划**：

   * 高/中权重 → micro 摘要；
   * 低权重 → 自适应块摘要；
   * 预算不够 → 一次性比例缩放；
   * 极端场景 → 全局块摘要。
5. **升级池**：将“高价值密度”消息升级为原文。
6. **并发生成摘要**：按缩放后的预算调用模型生成。
7. **双重护栏组装**：确保覆盖率可计算、无遗漏、可兜底。
8. **Top-up 填窗**：升级 micro → 加入未落地原文 → 达标 85%。
9. **保护当前用户消息**：保证其在最后且未被破坏。

---

## 事件--进度--统计

* **ProgressTracker** 会向前端发出阶段事件：开始 → 更新 → 完成。
* **漂亮进度条**：`[████▓░░░] 63.4%` 样式输出。
* **ProcessingStats**：

  * 内容：窗口使用率、覆盖率、保留原文/摘要条数、缓存命中、并发任务、API失败次数等；
  * 用途：排障、观测性能、验证覆盖效果。

---

## 常见问题

**Q1：为什么看起来“摘要很多”？**
A：Coverage-First 的特性——先覆盖所有历史，再用升级池把高价值消息升级为原文，最后用 Top-up 拉满窗口。

**Q2：真的不会丢内容吗？**
A：开启零丢失相关配置时，会尽量避免截断；同时护栏 B 会对未落地消息产出“简化摘要”兜底，覆盖率可在统计中看到。

**Q3：没有外部 API 也能跑吗？**
A：可以。会退化到轻量召回 + 启发式兜底；如需更强检索/摘要体验，建议配置向量与文本/视觉模型。

**Q4：为什么窗口利用率不到 85%？**
A：受回答缓冲区、模型安全系数、历史极大且“价值密度”不够等影响。可调高 `target_window_usage` 或放宽 `response_buffer_*`。

---

## 性能与调优建议

* 开启 **EmbeddingCache**，减小重复对话的向量开销。
* 合理调大 `max_concurrent_requests`（默认 6）以提升吞吐。
* 调整 `raw_block_target / max_blocks` 平衡“块摘要质量 vs. 速度”。
* 对“极长代码/日志”，适度提高 `chunk_target_tokens` 并增加 `chunk_overlap_tokens`。
* 将“高频模型”加入 `excluded_models`，只在需要的模型上启用插件。

---

## 迁移指南（从旧版本）

* **命名**：本版强调 *Zero-Loss Coverage-First*，与老的 “Multimodal Context Manager” 区分。
* **配置变更**：

  * 新增/强化 `upgrade_min_pct`（升级池保护）。
  * 新增/强化 `target_window_usage`（Top-up 目标）。
  * 自适应分块参数更细（`raw_block_target / max_blocks / floor_block`）。
* **行为差异**：

  * 先覆盖再优化，统计项更可见；
  * 覆盖失败会触发护栏 B 简化摘要，避免“无声丢失”。

---

## 更新日志

### 2.4.6（当前）

* Auto Memory 更健壮：用户无记忆返回 404 视为正常，不再误报。
* LLM 输出解析更稳：兼容非 JSON / code fence / 常见脏格式。
* OpenAI-compatible 增强：统一使用 `chat.completions.create`，不依赖 `.parse()`。
* Memory Update 兼容 `content / new_content` 字段别名。
* 记忆执行增加 fallback 与更直观日志，方便排障与验证“是否真的写入”。

### 2.4.5

* 全新 Coverage-First 规划：自适应分块 + 统一缩放 + 极端退化。
* 升级池保护：按价值密度贪心升级原文。
* 双重护栏：A（映射/范围校验）+ B（简化摘要兜底）。
* Top-up 填窗：升级 micro → 加原文，拉到目标使用率。
* 详尽统计与进度：覆盖率、利用率、并发、缓存命中、失败计数。

### 2.4.4 及更早

* 修复 ID 稳定性、窗口填充统计、数据 URI 校验、语法错误等。
* 多模态策略与图片识别模板完善。

---

# English Documentation

## Key Highlights

**v2.4.6** builds on **Coverage-First + Zero-Loss** and improves robustness & compatibility:

* **Coverage-First strategy**: cover history first, then optimize.
* **Adaptive block planning** based on raw tokens, role switches, time gaps, and score diffs.
* **One-shot proportional scaling** across micro & block summaries to respect budget.
* **Protected upgrade pool**: greedily promote high value-density items back to **verbatim**.
* **Dual-guard assembly**:

  * Guard A: mapping/range validation to avoid gaps;
  * Guard B: simplified fallback summaries to guarantee measurable coverage.
* **Top-up filler**: upgrade existing micros → add untouched verbatims to reach target window usage (85% by default).
* **Zero-loss mode (configurable)**: avoid truncation when possible + Guard-B fallbacks.
* **Progress & Stats**: phase progress bars; coverage, utilization, cache hits, concurrency, API failures, etc.
* **Multimodal strategies**: direct multimodal, vision-to-text, or multimodal vector RAG.
* **Broad model matcher**: GPT/Claude/Qwen/Doubao/GLM… with auto token limits & safety margin.
* **Vector cache & concurrency**: two-stage recall and parallel embeddings.
* **2.4.6 compatibility upgrades**:

  * Treat “no memories found (404)” as normal.
  * More tolerant parsing for non-JSON / fenced / messy LLM outputs.
  * Use `chat.completions.create` for OpenAI-compatible backends (no `.parse()` dependency).
  * Accept `content/new_content` alias for memory updates.
  * Extra fallback path & clearer logs for memory operations.

---

## Architecture Overview

```
Messages → Stable IDs → Multimodal Preprocess →
Two-stage Recall (lightweight → vectors) → Coverage Planning (micro + blocks) →
Proportional Scaling → Upgrade Pool → Parallel Summaries → Dual-Guard Assembly →
Top-up Filler → User-Message Protection → Output
```

Key modules: **MessageOrder**, **TokenCalculator**, **MessageChunker**, **CoveragePlanner**, **ProgressTracker**, **ProcessingStats**, **Top-up Filler**.

---

## Installation

In **Open WebUI**: Settings → Pipelines → **+** → paste the v2.4.6 source → Save.
Pin the pipeline as needed.

Dependencies are optional. Without external SDKs it gracefully degrades to local heuristics.

---

## Quick Start

Minimal configuration (defaults already work):

```yaml
enable_processing: true
token_safety_ratio: 0.92
target_window_usage: 0.85

enable_coverage_first: true
enable_zero_loss_guarantee: true
disable_insurance_truncation: true

debug_level: 0
```

Optional external APIs:

```yaml
api_base: "https://ark.cn-beijing.volces.com/api/v3"
api_key:  "<YOUR_API_KEY>"
text_model: "doubao-1-5-lite-32k-250115"
multimodal_model: "doubao-1.5-vision-pro-250328"
text_vector_model: "doubao-embedding-large-text-250515"
multimodal_vector_model: "doubao-embedding-vision-250615"
```

---

## Essential Config (Valves)

* **Window & Budget**: `token_safety_ratio`, `target_window_usage`, `response_buffer_*`
* **Coverage**: `coverage_*_threshold`, `coverage_*_summary_tokens`, `upgrade_min_pct`
* **Adaptive Blocks**: `raw_block_target`, `max_blocks`, `floor_block`
* **Multimodal**: `enable_multimodal`, `preserve_images_in_multimodal`, `always_process_images_before_summary`
* **Protection**: `force_preserve_current_user_message`, `preserve_recent_exchanges`, `disable_insurance_truncation`

> For the **full list**, see the `Filter.Valves` defaults in code.

---

## Pipeline Flow

1. Stable IDs & separation (system/history/current user).
2. Multimodal strategy selection.
3. Two-stage recall (lightweight → vectors, parallelized).
4. Coverage planning (micro for high/mid; adaptive blocks for low).
   Proportional scaling; global block fallback in extreme cases.
5. Upgrade pool (value-density greedy selection).
6. Parallel summary generation (respect scaled budgets).
7. Dual-guard assembly (A: mapping/range; B: simplified fallback).
8. Top-up window filling to reach target usage (≈85%).
9. Ensure the latest user message is preserved and last.

---

## Events--Progress--Stats

* Progress events for each phase with clean progress bars.
* Stats include coverage, utilization, preserved/summary counts, cache hits, concurrency, API failures, etc.

---

## FAQ

**Q: Why so many summaries?**
A: Coverage-First covers the whole history first; then the upgrade pool restores high-value messages to verbatim; finally Top-up fills unused space.

**Q: Is it truly lossless?**
A: With zero-loss options enabled, truncation is avoided when possible; Guard-B fallbacks ensure every uncovered item gets a simplified summary.

**Q: Can it run without external APIs?**
A: Yes, it gracefully falls back to lightweight recall and heuristics.

**Q: Why window usage < 85%?**
A: Response buffer, safety margin, or low value-density may limit fill; increase `target_window_usage` or relax `response_buffer_*`.

---

## Performance--Tuning

* Enable **EmbeddingCache** to reduce repeated vector costs.
* Increase `max_concurrent_requests` for higher throughput.
* Tune `raw_block_target / max_blocks` to balance quality vs. speed.
* For long code/logs, raise `chunk_target_tokens` and `chunk_overlap_tokens`.
* Use `excluded_models` to scope the pipeline.

---

## Migration Guide

* Renamed emphasis: **Zero-Loss Coverage-First** vs. old “Multimodal Context Manager”.
* New/stronger knobs: `upgrade_min_pct`, `target_window_usage`, refined adaptive-block params.
* Behavioral shift: measurable coverage via Guard-B fallbacks; more robust compatibility in 2.4.6.

---

## Changelog

### 2.4.6

* Robust Auto Memory handling for “no memories found (404)”.
* Tolerant LLM output parsing (non-JSON / fenced / messy formats).
* OpenAI-compatible improvements: always use `chat.completions.create` (no `.parse()` dependency).
* Memory update alias support: `content / new_content`.
* Extra fallback path + clearer logs for memory operations.

### 2.4.5

* Adaptive Coverage planning + proportional scaling + global fallback.
* Protected upgrade pool by value-density.
* Dual-guard assembly with simplified fallbacks.
* Top-up filler to reach target window usage.
* Extended stats & progress reporting.

### 2.4.4 and earlier

* Stable IDs, window-fill stats fixes, data-URI validation, syntax fixes.
* Improved multimodal strategies and vision prompts.

---

## License--Credits

**License:** MIT
**Author:** JiangNanGenius
Thanks to the Open WebUI community and contributors.

```
```
