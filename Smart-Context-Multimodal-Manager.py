"""
title: 🚀 高级上下文管理器（多模态+上下文最大化） - v2.6.3
author: JiangNanGenius
version: 2.6.3
license: MIT
required_open_webui_version: 0.5.17
author_url: https://github.com/JiangNanGenius
git_url: https://github.com/JiangNanGenius/Advanced-Contex-Manager-Multimodal
description: 高级上下文管理器（上下文最大化 + 多模态转写）+ 自动记忆（后台运行，不在前台显示状态）
"""

import json
import hashlib
import asyncio
import re
import base64
import math
import time
import copy
import html
import threading
import logging
import traceback
from datetime import datetime
from typing import (
    Optional,
    List,
    Dict,
    Callable,
    Any,
    Tuple,
    Union,
    Literal,
    cast,
    Type,
    TypeVar,
)
from pydantic import BaseModel, Field, AliasChoices, ValidationError, create_model
from enum import Enum
from collections import defaultdict

# Open WebUI相关导入
from fastapi import HTTPException, Request
from open_webui.main import app as webui_app
from open_webui.models.users import UserModel, Users
from open_webui.retrieval.vector.main import SearchResult
from open_webui.routers.memories import (
    AddMemoryForm,
    MemoryUpdateModel,
    QueryMemoryForm,
    add_memory,
    delete_memory_by_id,
    query_memory,
    update_memory_by_id,
)

# 导入依赖库
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

# ========== Auto Memory相关类定义 ==========


class Memory(BaseModel):
    """单个记忆条目"""

    mem_id: str = Field(..., description="记忆ID")
    created_at: datetime = Field(..., description="创建时间")
    update_at: datetime = Field(..., description="更新时间")
    content: str = Field(..., description="记忆内容")
    similarity_score: Optional[float] = Field(None, description="相似度分数")


class MemoryAddAction(BaseModel):
    action: Literal["add"] = Field(..., description="添加操作")
    content: str = Field(..., description="记忆内容")


class MemoryUpdateAction(BaseModel):
    action: Literal["update"] = Field(..., description="更新操作")
    id: str = Field(..., description="记忆ID")
    # 兼容字段名：既接受 content，也接受历史版本的 new_content
    content: str = Field(
        ...,
        description="新内容",
        validation_alias=AliasChoices("content", "new_content"),
    )


class MemoryDeleteAction(BaseModel):
    action: Literal["delete"] = Field(..., description="删除操作")
    id: str = Field(..., description="记忆ID")


class MemoryActionRequestStub(BaseModel):
    """记忆操作请求"""

    actions: list[Union[MemoryAddAction, MemoryUpdateAction, MemoryDeleteAction]] = (
        Field(
            default_factory=list,
            description="操作列表",
            max_length=20,
        )
    )
    # 仅用于调试/解释：即使 actions 为空，也请给出简短原因
    reason: str = Field(
        default="", description="Why actions is empty / rationale (debug)"
    )


# Memory系统提示词
UNIFIED_SYSTEM_PROMPT = """\
You manage a per-user Memory collection (short factual notes) to personalize future chats.

Inputs you will receive:
1) Recent conversation messages (displayed with negative indices; -1 is the most recent overall message).
   Usually -2 is the user's latest message.
2) A list of existing related memories (may be empty).

Goal:
Decide what actions to take on the memory collection. Focus primarily on the user's latest message, but you MAY use the
recent surrounding context to interpret intent (e.g., ongoing goals, preferences, or commitments). Prefer 0-2 actions.

Language policy (VERY IMPORTANT):
- Write memory "content" in the same language as the user's original wording. Do NOT translate.
- If the user's latest message is in Chinese, write the memory in Chinese; if in English, write in English.
- If the user mixes languages, keep the dominant language and preserve proper nouns / product names / code / quotes verbatim.
- Write "reason" in the same language as the user's latest message (debug only).

What to remember (good candidates):
- Stable preferences: language, tone, formatting, brevity, citation style, etc.
- Ongoing projects/goals/plans that will matter in future sessions.
- Repeated workflows/habits the user is actively adopting (even if expressed as a question that signals intent).
  Example: "How do I track progress with spaced repetition?" => user is using spaced repetition and wants progress tracking.
- Constraints that shape advice: device, environment, tools, budget ranges, recurring schedules, etc.
- Explicit requests like "remember this / please remember".

What NOT to remember:
- One-off questions that do not reveal stable preferences, commitments, or background.
- Temporary states or short-lived logistics.
- Secrets/credentials/IDs. Avoid sensitive personal data (health, politics, etc.) unless the user explicitly asks you to store it.

Actions:
- ADD: Create a new memory (1 short sentence, specific, neutral).
- UPDATE: Update an existing memory (by id) when the same fact changed/refined.
- DELETE: Delete a memory (by id) when user asks to forget, or when it is clearly obsolete/duplicated.

When actions are empty, still provide a short reason for debugging.

Follow the JSON output rules provided separately.

"""

AUTO_MEMORY_OUTPUT_INSTRUCTIONS = """\
Return ONLY a valid JSON object. Do not include markdown/code fences or any extra text.

Language:
- The values of "content" and "reason" MUST preserve the user's original language. Do NOT translate.
- If the latest user message is Chinese, output Chinese; if English, output English. If mixed, keep dominant language and preserve key terms verbatim.

Schema:
{
  "actions": [
    {"action":"add","content":"<memory text>"},
    {"action":"update","id":"<id>","content":"<memory text>"},
    {"action":"delete","id":"<id>"}
  ],
  "reason": "<short explanation for debugging (even when actions is empty)>"
}

Rules:
- Always include the top-level key "actions" (can be an empty list).
- Always include "reason" (can be empty string).
- If no actions are needed, return: {"actions": [], "reason": "..."}
- For update/delete, "id" MUST be one of the provided existing IDs.
"""


STRINGIFIED_MESSAGE_TEMPLATE = "-{index}. {role}: ```{content}```"


def searchresults_to_memories(results: SearchResult) -> list[Memory]:
    """将搜索结果转换为Memory对象"""
    memories = []
    if not results.ids or not results.documents or not results.metadatas:
        raise ValueError("SearchResult must contain ids, documents, and metadatas")

    for batch_idx, (ids_batch, docs_batch, metas_batch) in enumerate(
        zip(results.ids, results.documents, results.metadatas)
    ):
        distances_batch = results.distances[batch_idx] if results.distances else None
        for doc_idx, (mem_id, content, meta) in enumerate(
            zip(ids_batch, docs_batch, metas_batch)
        ):
            if not meta:
                raise ValueError(f"Missing metadata for memory id={mem_id}")
            if "created_at" not in meta:
                raise ValueError(
                    f"Missing 'created_at' in metadata for memory id={mem_id}"
                )
            if "updated_at" not in meta:
                meta["updated_at"] = meta["created_at"]

            created_at = datetime.fromtimestamp(meta["created_at"])
            updated_at = datetime.fromtimestamp(meta["updated_at"])

            similarity_score = None
            if distances_batch is not None and doc_idx < len(distances_batch):
                similarity_score = round(distances_batch[doc_idx], 3)

            mem = Memory(
                mem_id=mem_id,
                created_at=created_at,
                update_at=updated_at,
                content=content,
                similarity_score=similarity_score,
            )
            memories.append(mem)

    return memories


def build_actions_request_model(existing_ids: list[str]):
    """动态构建记忆操作请求模型"""
    if not existing_ids:
        allowed_actions = MemoryAddAction
    else:
        id_literal_type = Literal[tuple(existing_ids)]
        DynamicMemoryUpdateAction = create_model(
            "MemoryUpdateAction",
            id=(id_literal_type, ...),
            __base__=MemoryUpdateAction,
        )
        DynamicMemoryDeleteAction = create_model(
            "MemoryDeleteAction",
            id=(id_literal_type, ...),
            __base__=MemoryDeleteAction,
        )
        allowed_actions = Union[
            MemoryAddAction, DynamicMemoryUpdateAction, DynamicMemoryDeleteAction
        ]

    return create_model(
        "MemoriesActionRequest",
        actions=(
            list[allowed_actions],
            Field(
                default_factory=list,
                description="List of actions to perform on memories",
                max_length=20,
            ),
        ),
        reason=(
            str,
            Field(default="", description="Why actions is empty / rationale (debug)"),
        ),
        __base__=BaseModel,
    )


def _run_detached(
    coro, *, name: str = "detached", logger: Optional[logging.Logger] = None
):
    """在独立线程中运行协程（捕获异常并记录日志）"""

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        except Exception:
            if logger:
                logger.exception("Detached coroutine crashed: %s", name)
            else:
                traceback.print_exc()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_runner, daemon=True).start()


# ========== 上下文管理器相关类定义 ==========


class EmbeddingCache:
    """向量缓存器 - 基于content_key缓存"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get(self, content_key: str) -> Optional[List[float]]:
        """获取缓存的向量"""
        if content_key in self.cache:
            self.access_count[content_key] = self.access_count.get(content_key, 0) + 1
            return self.cache[content_key]
        return None

    def set(self, content_key: str, embedding: List[float]):
        """设置缓存的向量"""
        if len(self.cache) >= self.max_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_count[least_used[0]]
        self.cache[content_key] = embedding
        self.access_count[content_key] = 1

    def clear(self):
        """清理缓存"""
        self.cache.clear()
        self.access_count.clear()


class MessageOrder:
    """消息顺序管理器 - ID稳定化改进"""

    def __init__(self, original_messages: List[dict]):
        self.original_messages = original_messages
        self.order_map = {}
        self.message_ids = {}
        self.content_map = {}

        for i, msg in enumerate(self.original_messages):
            content_key = self._generate_stable_content_key(msg)
            msg_id = hashlib.md5(f"{i}_{content_key}".encode()).hexdigest()[:16]
            self.order_map[msg_id] = i
            self.message_ids[i] = msg_id
            self.content_map[content_key] = i

            msg["_order_id"] = msg_id
            msg["_original_index"] = i
            msg["_content_key"] = content_key

    def _generate_stable_content_key(self, msg: dict) -> str:
        """生成稳定的消息内容标识"""
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            content_parts = []
            for item in content:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", "")[:100])
                elif item.get("type") == "image_url":
                    image_data = item.get("image_url", {}).get("url", "")
                    if image_data.startswith("data:"):
                        try:
                            header, data = image_data.split("base64,", 1)
                            content_parts.append(f"[IMAGE:{header}:{data[:50]}]")
                        except:
                            content_parts.append("[IMAGE:invalid]")
                    else:
                        content_parts.append(f"[IMAGE:url:{image_data[:50]}]")
            content_str = " ".join(content_parts)
        else:
            content_str = str(content)[:200]

        return f"{role}:{content_str}"

    def generate_chunk_id(self, msg_id: str, chunk_index: int) -> str:
        """生成chunk ID"""
        return f"{msg_id}#{chunk_index}"

    def find_current_user_message_index(self, messages: List[dict]) -> int:
        """找到当前用户消息的索引"""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                return i
        return -1

    def sort_messages_preserve_user(
        self, messages: List[dict], current_user_message: dict = None
    ) -> List[dict]:
        """根据原始顺序排序消息，保护当前用户消息位置"""
        if not messages:
            return messages

        other_messages = []
        current_user_in_list = None

        for msg in messages:
            if current_user_message and msg.get(
                "_order_id"
            ) == current_user_message.get("_order_id"):
                current_user_in_list = msg
            else:
                other_messages.append(msg)

        def get_order(msg):
            return msg.get("_original_index", 999999)

        other_messages.sort(key=get_order)

        if current_user_in_list:
            return other_messages + [current_user_in_list]
        else:
            return other_messages

    def get_message_preview(self, msg: dict) -> str:
        """获取消息预览用于调试"""
        if isinstance(msg.get("content"), list):
            text_parts = []
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append("[图片]")
            content = " ".join(text_parts)
        else:
            content = str(msg.get("content", ""))

        content = content.replace("\n", " ").replace("\r", " ")
        content = re.sub(r"\s+", " ", content).strip()
        return content[:100] + "..." if len(content) > 100 else content


class ProcessingStats:
    """处理统计信息记录器"""

    def __init__(self):
        # 基础统计
        self.original_tokens = 0
        self.original_messages = 0
        self.final_tokens = 0
        self.final_messages = 0
        self.token_limit = 0
        self.target_tokens = 0
        self.current_user_tokens = 0

        # 处理统计
        self.iterations = 0
        self.chunked_messages = 0
        self.summarized_messages = 0
        self.vector_retrievals = 0
        self.rerank_operations = 0
        self.multimodal_processed = 0
        self.processing_time = 0.0

        # 上下文最大化策略统计
        self.coverage_rate = 0.0
        self.coverage_total_messages = 0
        self.coverage_preserved_count = 0
        self.coverage_preserved_tokens = 0
        self.coverage_summary_count = 0
        self.coverage_summary_tokens = 0
        self.coverage_micro_summaries = 0
        self.coverage_block_summaries = 0
        self.coverage_upgrade_count = 0
        self.coverage_upgrade_tokens_saved = 0
        self.coverage_budget_usage = 0.0

        # 分块与预算统计
        self.chunked_messages_count = 0
        self.total_chunks_created = 0
        self.adaptive_blocks_created = 0
        self.block_merge_operations = 0
        self.budget_scaling_applied = 0
        self.scaling_factor = 1.0

        # 护栏统计
        self.guard_a_warnings = 0
        self.guard_b_fallbacks = 0
        self.id_mapping_errors = 0

        # 不截断保障统计
        self.zero_loss_guarantee = True
        self.budget_adjustments = 0
        self.min_budget_applied = 0
        self.insurance_truncation_avoided = 0

        # Top-up统计
        self.topup_applied = 0
        self.topup_micro_upgraded = 0
        self.topup_raw_added = 0
        self.topup_tokens_added = 0

        # 性能统计
        self.api_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.concurrent_tasks = 0
        self.embedding_requests = 0
        self.summary_requests = 0

        # 其他统计
        self.preserved_messages = 0
        self.processed_messages = 0
        self.summary_messages = 0
        self.emergency_truncations = 0
        self.content_loss_ratio = 0.0
        self.discarded_messages = 0
        self.recovered_messages = 0
        self.window_utilization = 0.0
        self.try_preserve_tokens = 0
        self.try_preserve_messages = 0
        self.try_preserve_summary_messages = 0
        self.keyword_generations = 0
        self.context_maximization_detections = 0
        self.chunk_created = 0
        self.chunk_processed = 0
        self.recursive_summaries = 0
        self.context_max_direct_preserve = 0
        self.context_max_chunked = 0
        self.context_max_summarized = 0
        self.multimodal_extracted = 0
        self.fallback_preserve_applied = 0
        self.user_message_recovery_count = 0
        self.rag_no_results_count = 0
        self.history_message_separation_count = 0
        self.image_processing_errors = 0
        self.syntax_errors_fixed = 0
        self.truncation_skip_count = 0
        self.truncation_recovered_messages = 0
        self.smart_truncation_applied = 0

    def calculate_retention_ratio(self) -> float:
        """计算内容保留比例"""
        if self.original_tokens == 0:
            return 0.0
        return self.final_tokens / self.original_tokens

    def calculate_window_usage_ratio(self) -> float:
        """计算对话窗口使用率"""
        if self.target_tokens == 0:
            return 0.0
        return self.final_tokens / self.target_tokens

    def get_summary(self) -> str:
        """获取统计摘要"""
        retention = self.calculate_retention_ratio()
        window_usage = self.calculate_window_usage_ratio()
        status = "✅" if self.zero_loss_guarantee else "⚠️"
        summary_lines = [
            "📊 上下文最大化处理完成:",
            f"├─ 消息: {self.original_messages} -> {self.final_messages}条 | tokens: {self.original_tokens:,} -> {self.final_tokens:,}",
            f"├─ 窗口使用: {window_usage:.1%} | 内容保留: {retention:.1%}",
            f"├─ Coverage: 覆盖率{self.coverage_rate:.1%}, 原文{self.coverage_preserved_count}条+摘要{self.coverage_summary_count}条",
            f"├─ 性能: 处理{self.processing_time:.1f}s, API调用{self.summary_requests}次, 缓存命中{self.cache_hits}次",
            f"└─ 不截断: {status}",
        ]
        return "\n".join(summary_lines)


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, event_emitter):
        self.event_emitter = event_emitter
        self.current_step = 0
        self.total_steps = 0
        self.current_phase = ""
        self.phase_progress = 0
        self.phase_total = 0
        self.logged_phases = set()

    def create_progress_bar(self, percentage: float, width: int = 15) -> str:
        """创建美观的进度条"""
        filled = int(percentage * width / 100)
        if percentage >= 100:
            bar = "█" * width
        else:
            bar = "█" * filled + "▓" * max(0, 1) + "░" * max(0, width - filled - 1)
        return f"[{bar}] {percentage:.1f}%"

    async def start_phase(self, phase_name: str, total_items: int = 0):
        """开始新阶段"""
        self.current_phase = phase_name
        self.phase_progress = 0
        self.phase_total = total_items
        self.logged_phases.add(phase_name)
        await self.update_status(f"开始 {phase_name}")

    async def update_progress(
        self, completed: int, total: int = None, detail: str = ""
    ):
        """更新进度"""
        if total is None:
            total = self.phase_total
        self.phase_progress = completed

        if total > 0:
            percentage = (completed / total) * 100
            progress_bar = self.create_progress_bar(percentage)
            status = f"{self.current_phase} {progress_bar}"
            if detail:
                status += f" - {detail}"
        else:
            status = f"{self.current_phase}"
            if detail:
                status += f" - {detail}"

        await self.update_status(status, False)

    async def complete_phase(self, message: str = ""):
        """完成当前阶段"""
        final_message = f"{self.current_phase} 完成"
        if message:
            final_message += f" - {message}"
        await self.update_status(final_message, True)

    async def update_status(self, message: str, done: bool = False):
        """更新状态"""
        if self.event_emitter:
            try:
                message = message.replace("\n", " ").replace("\r", " ")
                message = re.sub(r"\s+", " ", message).strip()
                await self.event_emitter(
                    {
                        "type": "status",
                        "data": {"description": message, "done": done},
                    }
                )
            except Exception as e:
                if str(e) not in self.logged_phases:
                    print(f"⚠️ 进度更新失败: {e}")
                    self.logged_phases.add(str(e))


class ModelMatcher:
    """智能模型匹配器"""

    def __init__(self):
        self.default_limit = 200000
        self.default_image_tokens = 1500

    def _build_model_info(
        self,
        family: str,
        multimodal: bool,
        limit: int,
        image_tokens: int,
        match_type: str,
        matched_pattern: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "family": family,
            "multimodal": multimodal,
            "limit": limit,
            "image_tokens": image_tokens,
            "match_type": match_type,
        }
        if matched_pattern:
            result["matched_pattern"] = matched_pattern
        if hint:
            result["hint"] = hint
        return result

    def match_model(self, model_name: str) -> Dict[str, Any]:
        """智能匹配模型信息"""
        if not model_name:
            return self._build_model_info(
                family="unknown",
                multimodal=False,
                limit=self.default_limit,
                image_tokens=self.default_image_tokens,
                match_type="default",
            )

        model_lower = model_name.lower().strip()

        if re.match(r"gpt-5.*", model_lower):
            return self._build_model_info("gpt", True, 200000, 2000, "fuzzy", "gpt-5.*")
        if re.match(r"gpt-4o.*", model_lower):
            return self._build_model_info(
                "gpt", True, 128000, 1500, "fuzzy", "gpt-4o.*"
            )
        if re.match(r"gpt-4.*", model_lower):
            return self._build_model_info("gpt", False, 8192, 0, "fuzzy", "gpt-4.*")
        if re.match(r"claude-4.*", model_lower):
            return self._build_model_info(
                "claude", True, 200000, 1000, "fuzzy", "claude-4.*"
            )
        if re.match(r"claude-3.*", model_lower):
            return self._build_model_info(
                "claude", True, 200000, 1000, "fuzzy", "claude-3.*"
            )
        if re.match(r"doubao.*vision.*", model_lower):
            return self._build_model_info(
                "doubao", True, 128000, 1500, "fuzzy", "doubao.*vision.*"
            )
        if re.match(r"doubao.*", model_lower):
            return self._build_model_info(
                "doubao", False, 50000, 0, "fuzzy", "doubao.*"
            )
        if re.match(r"gemini.*vision.*", model_lower):
            return self._build_model_info(
                "gemini", True, 128000, 800, "fuzzy", "gemini.*vision.*"
            )
        if re.match(r"qwen.*vl.*", model_lower):
            return self._build_model_info(
                "qwen", True, 32000, 1000, "fuzzy", "qwen.*vl.*"
            )

        return self._build_model_info(
            family="unknown",
            multimodal=False,
            limit=self.default_limit,
            image_tokens=self.default_image_tokens,
            match_type="default",
            hint=f"未识别模型 '{model_name}'，已使用默认参数（200k / 文本模型）。",
        )


class TokenCalculator:
    """Token计算器"""

    def __init__(self):
        self._encoding = None
        self.model_info = None

    def set_model_info(self, model_info: dict):
        """设置当前模型信息"""
        self.model_info = model_info

    def get_encoding(self):
        """获取tiktoken编码器"""
        if not TIKTOKEN_AVAILABLE:
            return None
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """简化的token计算"""
        if not text:
            return 0
        encoding = self.get_encoding()
        if encoding:
            try:
                return len(encoding.encode(str(text)))
            except Exception:
                pass
        return len(str(text)) // 4

    def calculate_image_tokens(self, image_data: str) -> int:
        """计算图片tokens"""
        if self.model_info:
            return self.model_info.get("image_tokens", 1500)
        return 1500


class InputCleaner:
    """输入清洗与严格兜底"""

    @staticmethod
    def clean_text_for_regex(text: str) -> str:
        """清洗文本用于正则表达式"""
        if not text:
            return ""
        try:
            text = text.replace("\u2028", " ").replace("\u2029", " ")
            text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
            text = text.replace("\n", " ").replace("\r", " ")
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            print(f"⚠️ 文本清理异常: {str(e)[:100]}")
            return "".join(c for c in str(text) if c.isprintable() or c.isspace())[
                :1000
            ]

    @staticmethod
    def validate_and_clean_image_url(image_url: str) -> Tuple[bool, str]:
        """验证并清洗图片URL"""
        if not image_url or not isinstance(image_url, str):
            return False, ""

        try:
            image_url = image_url.strip()

            if image_url.startswith(("http://", "https://")):
                return True, image_url

            if image_url.startswith("data:"):
                if "base64," not in image_url:
                    return False, ""
                header, b64 = image_url.split("base64,", 1)
                if not header.lower().startswith("data:image/"):
                    return False, ""
                b64_str = re.sub(r"\s+", "", b64)
                if len(b64_str) < 100:
                    return False, ""
                head = b64_str[:100]
                pad_len = (-len(head)) % 4
                try:
                    base64.b64decode(head + ("=" * pad_len), validate=True)
                except Exception:
                    return False, ""
                return True, f"{header}base64,{b64_str}"

            return False, ""
        except Exception as e:
            print(f"⚠️ 图片URL验证异常: {str(e)[:100]}")
            return False, ""

    @staticmethod
    def safe_regex_match(pattern: str, text: str) -> bool:
        """安全的正则匹配"""
        try:
            cleaned_text = InputCleaner.clean_text_for_regex(text)
            return re.search(pattern, cleaned_text) is not None
        except Exception as e:
            print(f"⚠️ 正则匹配异常: {str(e)[:100]}")
            return False


class MessageChunker:
    """单条消息内分片处理器"""

    def __init__(self, token_calculator: TokenCalculator, valves):
        self.token_calculator = token_calculator
        self.valves = valves

    def should_chunk_message(self, message: dict) -> bool:
        """判断消息是否需要分片"""
        tokens = self.token_calculator.count_tokens(self.extract_text_content(message))
        return tokens > self.valves.large_message_threshold

    def extract_text_content(self, message: dict) -> str:
        """从消息中提取文本内容"""
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append("[图片]")
            return " ".join(text_parts)
        else:
            return str(content)

    def chunk_single_message(
        self, message: dict, message_order: MessageOrder
    ) -> List[dict]:
        """对单条消息进行分片处理"""
        content_text = self.extract_text_content(message)
        if not self.should_chunk_message(message):
            return [message]

        chunks = self._intelligent_chunk_text(content_text)
        if len(chunks) <= 1:
            return [message]

        chunked_messages = []
        msg_id = message.get("_order_id", "unknown")
        for i, chunk_text in enumerate(chunks):
            chunk_id = message_order.generate_chunk_id(msg_id, i)
            chunk_message = copy.deepcopy(message)
            chunk_message["content"] = chunk_text
            chunk_message["_order_id"] = chunk_id
            chunk_message["_is_chunk"] = True
            chunk_message["_parent_msg_id"] = msg_id
            chunk_message["_chunk_index"] = i
            chunk_message["_total_chunks"] = len(chunks)
            chunked_messages.append(chunk_message)

        return chunked_messages

    def _intelligent_chunk_text(self, text: str) -> List[str]:
        """智能文本分片"""
        if not text:
            return [text]

        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        text = text.replace("\u2028", "\n").replace("\u2029", "\n")

        target_size = self.valves.chunk_target_tokens * 4
        min_size = self.valves.chunk_min_tokens * 4
        max_size = self.valves.chunk_max_tokens * 4
        overlap_size = self.valves.chunk_overlap_tokens * 4

        chunks = []
        current_chunk = ""
        paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) > target_size and current_chunk:
                if len(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                    if self.valves.chunk_overlap_tokens > 0:
                        overlap_text = (
                            current_chunk[-overlap_size:]
                            if len(current_chunk) > overlap_size
                            else current_chunk
                        )
                        current_chunk = overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

            if len(current_chunk) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

        if current_chunk and len(current_chunk.strip()) >= min_size // 2:
            chunks.append(current_chunk.strip())
        elif current_chunk and chunks:
            chunks[-1] += "\n\n" + current_chunk.strip()
        elif current_chunk:
            chunks.append(current_chunk.strip())

        if not chunks and text:
            chunks = [text]

        return chunks

    def preprocess_messages_with_chunking(
        self, messages: List[dict], message_order: MessageOrder
    ) -> List[dict]:
        """预处理消息：对大消息进行分片"""
        processed_messages = []
        chunked_count = 0
        for message in messages:
            if self.should_chunk_message(message):
                chunked_messages = self.chunk_single_message(message, message_order)
                processed_messages.extend(chunked_messages)
                if len(chunked_messages) > 1:
                    chunked_count += 1
            else:
                processed_messages.append(message)
        return processed_messages


class CoveragePlanner:
    """Coverage计划器"""

    def __init__(self, token_calculator: TokenCalculator, valves):
        self.token_calculator = token_calculator
        self.valves = valves

    def plan_adaptive_coverage_summaries(
        self, scored_msgs: List[dict], total_budget: int
    ) -> Tuple[List[dict], int]:
        """规划自适应覆盖摘要"""
        if not scored_msgs:
            return [], 0

        HIGH, MID, LOW = self._classify_messages_by_score(scored_msgs)
        adaptive_blocks = self._create_adaptive_blocks(LOW)
        entries, ideal_total_cost = self._calculate_ideal_budgets(
            HIGH, MID, adaptive_blocks
        )

        if ideal_total_cost > total_budget:
            entries, actual_cost = self._apply_proportional_scaling(
                entries, total_budget
            )
        else:
            entries, actual_cost = self._apply_upward_expansion(
                entries, total_budget, ideal_total_cost
            )

        if actual_cost > total_budget * 1.1:
            entries, actual_cost = self._apply_extreme_fallback(
                scored_msgs, total_budget
            )

        return entries, actual_cost

    def _classify_messages_by_score(
        self, scored_msgs: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """按分数分档消息"""
        HIGH, MID, LOW = [], [], []
        for item in scored_msgs:
            if item["score"] >= self.valves.coverage_high_score_threshold:
                HIGH.append(item)
            elif item["score"] >= self.valves.coverage_mid_score_threshold:
                MID.append(item)
            else:
                LOW.append(item)
        return HIGH, MID, LOW

    def _create_adaptive_blocks(self, low_messages: List[dict]) -> List[dict]:
        """按原文token量自适应分块"""
        if not low_messages:
            return []

        low_sorted = sorted(low_messages, key=lambda x: x["idx"])
        blocks = []
        current_block = []
        current_tokens = 0
        raw_block_target = self.valves.raw_block_target

        for item in low_sorted:
            msg_tokens = item["tokens"]
            should_cut_block = False

            if current_tokens + msg_tokens > raw_block_target and current_block:
                should_cut_block = True

            if current_block and abs(item["idx"] - current_block[-1]["idx"]) > 5:
                should_cut_block = True

            if current_block:
                prev_role = current_block[-1]["msg"].get("role", "")
                curr_role = item["msg"].get("role", "")
                if (
                    prev_role != curr_role
                    and prev_role in ["user", "assistant"]
                    and curr_role in ["user", "assistant"]
                ):
                    should_cut_block = True

            if current_block:
                score_diff = abs(item["score"] - current_block[-1]["score"])
                if score_diff > 0.3:
                    should_cut_block = True

            if should_cut_block:
                if current_block:
                    blocks.append(
                        {
                            "type": "adaptive_block",
                            "idx_range": (
                                current_block[0]["idx"],
                                current_block[-1]["idx"],
                            ),
                            "msgs": [item["msg"] for item in current_block],
                            "raw_tokens": current_tokens,
                            "avg_score": sum(item["score"] for item in current_block)
                            / len(current_block),
                            "msg_count": len(current_block),
                        }
                    )
                current_block = [item]
                current_tokens = msg_tokens
            else:
                current_block.append(item)
                current_tokens += msg_tokens

        if current_block:
            blocks.append(
                {
                    "type": "adaptive_block",
                    "idx_range": (current_block[0]["idx"], current_block[-1]["idx"]),
                    "msgs": [item["msg"] for item in current_block],
                    "raw_tokens": current_tokens,
                    "avg_score": sum(item["score"] for item in current_block)
                    / len(current_block),
                    "msg_count": len(current_block),
                }
            )

        if len(blocks) > self.valves.max_blocks:
            blocks = self._merge_small_blocks(blocks)

        return blocks

    def _merge_small_blocks(self, blocks: List[dict]) -> List[dict]:
        """合并小块"""
        if len(blocks) <= self.valves.max_blocks:
            return blocks

        blocks.sort(key=lambda x: x["raw_tokens"])
        merged_blocks = []
        i = 0

        while i < len(blocks):
            current_block = blocks[i]
            if (
                i + 1 < len(blocks)
                and len(merged_blocks) + (len(blocks) - i) > self.valves.max_blocks
            ):
                next_block = blocks[i + 1]
                if (
                    current_block["raw_tokens"] + next_block["raw_tokens"]
                    <= self.valves.raw_block_target * 2
                ):
                    merged_block = {
                        "type": "adaptive_block",
                        "idx_range": (
                            current_block["idx_range"][0],
                            next_block["idx_range"][1],
                        ),
                        "msgs": current_block["msgs"] + next_block["msgs"],
                        "raw_tokens": current_block["raw_tokens"]
                        + next_block["raw_tokens"],
                        "avg_score": (
                            current_block["avg_score"] * current_block["msg_count"]
                            + next_block["avg_score"] * next_block["msg_count"]
                        )
                        / (current_block["msg_count"] + next_block["msg_count"]),
                        "msg_count": current_block["msg_count"]
                        + next_block["msg_count"],
                    }
                    merged_blocks.append(merged_block)
                    i += 2
                    continue
            merged_blocks.append(current_block)
            i += 1

        return merged_blocks

    def _calculate_ideal_budgets(
        self, high_msgs: List[dict], mid_msgs: List[dict], adaptive_blocks: List[dict]
    ) -> Tuple[List[dict], int]:
        """计算理想预算需求"""
        entries = []
        total_cost = 0

        for grp, per_token in [
            (high_msgs, self.valves.coverage_high_summary_tokens),
            (mid_msgs, self.valves.coverage_mid_summary_tokens),
        ]:
            for item in grp:
                msg_id = item["msg"].get("_order_id", f"msg_{item['idx']}")
                entry = {
                    "type": "micro",
                    "msg_id": msg_id,
                    "ideal_budget": per_token,
                    "floor_budget": max(self.valves.min_summary_tokens, per_token // 3),
                    "msg": item["msg"],
                    "score": item["score"],
                }
                entries.append(entry)
                total_cost += per_token

        for block in adaptive_blocks:
            floor_budget = max(
                self.valves.min_block_summary_tokens, self.valves.floor_block
            )
            size_factor = min(3.0, block["raw_tokens"] / self.valves.raw_block_target)
            ideal_budget = int(
                floor_budget
                + (self.valves.coverage_block_summary_tokens - floor_budget)
                * size_factor
            )
            block_key = f"block_{block['idx_range'][0]}_{block['idx_range'][1]}"
            entry = {
                "type": "adaptive_block",
                "block_key": block_key,
                "idx_range": block["idx_range"],
                "ideal_budget": ideal_budget,
                "floor_budget": floor_budget,
                "msgs": block["msgs"],
                "raw_tokens": block["raw_tokens"],
                "avg_score": block["avg_score"],
            }
            entries.append(entry)
            total_cost += ideal_budget

        return entries, total_cost

    def _apply_proportional_scaling(
        self, entries: List[dict], available_budget: int
    ) -> Tuple[List[dict], int]:
        """一次性比例缩放"""
        total_floors = sum(entry["floor_budget"] for entry in entries)
        total_ideals = sum(entry["ideal_budget"] for entry in entries)

        if total_floors > available_budget:
            return self._apply_extreme_fallback_from_entries(entries, available_budget)

        available_for_scaling = available_budget - total_floors
        scalable_amount = total_ideals - total_floors

        if scalable_amount <= 0:
            alpha = 0
        else:
            alpha = available_for_scaling / scalable_amount
            alpha = min(1.0, alpha)

        total_assigned = 0
        for entry in entries:
            floor_budget = entry["floor_budget"]
            ideal_budget = entry["ideal_budget"]
            scaled_budget = floor_budget + alpha * (ideal_budget - floor_budget)
            entry["budget"] = int(round(scaled_budget))
            total_assigned += entry["budget"]

        error = available_budget - total_assigned
        if error != 0:
            scored_entries = [
                (entry.get("score", entry.get("avg_score", 0)), entry)
                for entry in entries
            ]
            scored_entries.sort(key=lambda x: x[0], reverse=True)

            if error > 0:
                for _, entry in scored_entries:
                    if error <= 0:
                        break
                    entry["budget"] += 1
                    error -= 1
            else:
                for _, entry in reversed(scored_entries):
                    if error >= 0:
                        break
                    if entry["budget"] > entry["floor_budget"]:
                        entry["budget"] -= 1
                        error += 1

        final_cost = sum(entry["budget"] for entry in entries)
        return entries, final_cost

    def _apply_upward_expansion(
        self, entries: List[dict], available_budget: int, ideal_total_cost: int
    ) -> Tuple[List[dict], int]:
        """向上扩张模式"""
        expansion_cap = 3.0
        target_usage = self.valves.target_window_usage
        target_cost = int(available_budget * target_usage)

        if ideal_total_cost >= target_cost:
            for entry in entries:
                entry["budget"] = entry["ideal_budget"]
            return entries, ideal_total_cost

        expansion_factor = min(expansion_cap, target_cost / ideal_total_cost)
        total_assigned = 0

        for entry in entries:
            base_budget = entry["ideal_budget"]
            if entry["type"] == "adaptive_block":
                expanded_budget = int(base_budget * expansion_factor)
            elif (
                entry["type"] == "micro"
                and entry.get("score", 0) >= self.valves.coverage_high_score_threshold
            ):
                expanded_budget = int(base_budget * min(2.0, expansion_factor))
            else:
                expanded_budget = base_budget
            entry["budget"] = expanded_budget
            total_assigned += expanded_budget

        if total_assigned > available_budget:
            scale_down = available_budget / total_assigned
            for entry in entries:
                entry["budget"] = int(entry["budget"] * scale_down)
            total_assigned = sum(entry["budget"] for entry in entries)

        return entries, total_assigned

    def _apply_extreme_fallback(
        self, scored_msgs: List[dict], available_budget: int
    ) -> Tuple[List[dict], int]:
        """极端退化：单条全局块摘要"""
        global_budget = max(
            self.valves.min_block_summary_tokens, int(available_budget * 0.9)
        )
        sorted_msgs = sorted(scored_msgs, key=lambda x: x["idx"])
        all_msgs = [item["msg"] for item in sorted_msgs]
        entry = {
            "type": "global_block",
            "block_key": f"global_0_{len(sorted_msgs)-1}",
            "idx_range": (0, len(sorted_msgs) - 1),
            "budget": global_budget,
            "msgs": all_msgs,
            "avg_score": sum(item["score"] for item in sorted_msgs) / len(sorted_msgs),
        }
        return [entry], global_budget

    def _apply_extreme_fallback_from_entries(
        self, entries: List[dict], available_budget: int
    ) -> Tuple[List[dict], int]:
        """从现有条目执行极端退化"""
        all_msgs = []
        for entry in entries:
            if entry["type"] == "micro":
                all_msgs.append(entry["msg"])
            elif entry["type"] == "adaptive_block":
                all_msgs.extend(entry["msgs"])

        all_msgs.sort(key=lambda x: x.get("_original_index", 0))
        global_budget = max(
            self.valves.min_block_summary_tokens, int(available_budget * 0.9)
        )
        entry = {
            "type": "global_block",
            "block_key": f"global_0_{len(all_msgs)-1}",
            "idx_range": (0, len(all_msgs) - 1),
            "budget": global_budget,
            "msgs": all_msgs,
            "avg_score": 0.5,
        }
        return [entry], global_budget


# ========== 主过滤器类 ==========


class Filter:
    class Valves(BaseModel):
        # ========== Auto Memory配置 ==========
        enable_auto_memory: bool = Field(
            default=True, description="🧠 启用自动记忆管理"
        )
        memory_messages_to_consider: int = Field(
            default=4, description="🧠 记忆提取考虑的消息数"
        )
        memory_related_memories_n: int = Field(
            default=5, description="🧠 相关记忆检索数量"
        )
        memory_minimum_similarity: Optional[float] = Field(
            default=0.0, description="🧠 记忆最小相似度阈值"
        )
        memory_force_add_prefixes: str = Field(
            default="记住:;remember:",
            description="🧠 强制写入记忆前缀（用 ; 分隔），命中则直接 add，不经过LLM，例如：记住:;remember:",
        )
        override_memory_context: bool = Field(
            default=False, description="🧠 拦截并覆盖记忆上下文注入"
        )
        memory_show_status: bool = Field(
            default=False,
            description="🧠（已废弃）前台不显示自动记忆状态（保留字段仅为兼容旧配置）",
        )

        # ========== 基础控制 ==========
        enable_processing: bool = Field(
            default=True, description="🔄 启用内容最大化处理"
        )
        excluded_models: str = Field(
            default="", description="🚫 排除模型列表(逗号分隔)"
        )
        suppress_frontend_when_idle: bool = Field(
            default=True, description="🕶️ 无需处理时不显示任何前端进度/日志"
        )
        enable_window_topup: bool = Field(
            default=False, description="🧯 仅在超限压缩后才允许窗口填充"
        )

        # ========== 核心配置 ==========
        max_window_utilization: float = Field(
            default=0.95, description="🪟 最大窗口利用率(95%)"
        )
        aggressive_content_recovery: bool = Field(
            default=True, description="🔄 激进内容合并模式"
        )
        min_preserve_ratio: float = Field(
            default=0.75, description="🔒 最小内容保留比例(75%)"
        )

        # ========== 上下文最大化策略配置 ==========
        enable_coverage_first: bool = Field(
            default=True, description="🎯 启用上下文最大化策略"
        )
        coverage_high_score_threshold: float = Field(
            default=0.7, description="🎯 高权重阈值(70%)"
        )
        coverage_mid_score_threshold: float = Field(
            default=0.4, description="🎯 中权重阈值(40%)"
        )
        coverage_high_summary_tokens: int = Field(
            default=100, description="📄 高权重消息微摘要目标tokens"
        )
        coverage_mid_summary_tokens: int = Field(
            default=50, description="📄 中权重消息微摘要目标tokens"
        )
        coverage_low_summary_tokens: int = Field(
            default=20, description="📄 低权重消息微摘要目标tokens"
        )
        coverage_block_summary_tokens: int = Field(
            default=350, description="📚 块摘要目标tokens"
        )
        coverage_upgrade_ratio: float = Field(
            default=0.3, description="⬆️ 升级预算比例(30%)"
        )

        # ========== 自适应分块配置 ==========
        raw_block_target: int = Field(
            default=15000, description="🧩 自适应块目标原文tokens"
        )
        floor_block: int = Field(default=300, description="📏 块摘要最小预算tokens")
        max_blocks: int = Field(default=8, description="📚 最大块数量")
        upgrade_min_pct: float = Field(
            default=0.2, description="⬆️ 升级池最小预留比例(20%)"
        )

        # ========== 不截断保障配置 ==========
        enable_zero_loss_guarantee: bool = Field(
            default=True, description="🛡️ 启用不截断保障"
        )
        min_summary_tokens: int = Field(
            default=30, description="📏 最小微摘要tokens(保底)"
        )
        min_block_summary_tokens: int = Field(
            default=200, description="📏 最小块摘要tokens(保底)"
        )
        max_budget_adjustment_rounds: int = Field(
            default=5, description="🔧 最大预算调整轮次"
        )
        disable_insurance_truncation: bool = Field(
            default=True, description="🚫 禁用保险截断(强制不截断)"
        )

        # ========== 尽量保留配置 ==========
        enable_try_preserve: bool = Field(
            default=True, description="🔒 启用尽量保留机制"
        )
        try_preserve_ratio: float = Field(
            default=0.40, description="🔒 尽量保留预算比例(40%)"
        )
        try_preserve_exchanges: int = Field(
            default=3, description="🔒 尽量保留对话轮次数"
        )

        # ========== 响应空间配置 ==========
        response_buffer_ratio: float = Field(
            default=0.06, description="📝 响应空间预留比例(6%)"
        )
        response_buffer_max: int = Field(
            default=3000, description="📝 响应空间最大值(tokens)"
        )
        response_buffer_min: int = Field(
            default=1000, description="📝 响应空间最小值(tokens)"
        )

        # ========== 多模态处理配置 ==========
        multimodal_direct_threshold: float = Field(
            default=0.70, description="🎯 多模态直接输入Token预算阈值(70%)"
        )
        preserve_images_in_multimodal: bool = Field(
            default=True, description="📸 多模态模型是否保留原始图片"
        )
        always_process_images_before_summary: bool = Field(
            default=True, description="📝 摘要前总是先处理图片"
        )

        # ========== 上下文最大化处理配置 ==========
        enable_context_maximization: bool = Field(
            default=True, description="📚 启用上下文最大化处理"
        )
        context_max_direct_preserve_ratio: float = Field(
            default=0.40, description="📚 上下文最大化直接保留比例(40%)"
        )
        context_max_processing_ratio: float = Field(
            default=0.45, description="📚 上下文最大化处理预算比例(45%)"
        )
        context_max_fallback_ratio: float = Field(
            default=0.15, description="📚 上下文最大化容错预算比例(15%)"
        )
        context_max_skip_rag: bool = Field(
            default=True, description="📚 上下文最大化跳过RAG处理"
        )
        context_max_prioritize_recent: bool = Field(
            default=True, description="📚 上下文最大化优先保留最近内容"
        )

        # ========== 容错机制配置 ==========
        enable_fallback_preservation: bool = Field(
            default=True, description="🛡️ 启用容错保护机制"
        )
        fallback_preserve_ratio: float = Field(
            default=0.25, description="🛡️ 容错保护预留比例(25%)"
        )
        min_history_messages: int = Field(default=8, description="🛡️ 最少历史消息数量")
        force_preserve_recent_user_exchanges: int = Field(
            default=3, description="🛡️ 强制保留最近用户对话轮次"
        )

        # ========== 功能开关 ==========
        enable_multimodal: bool = Field(default=True, description="🖼️ 启用多模态处理")
        enable_vision_preprocessing: bool = Field(
            default=True, description="👁️ 启用图片预处理"
        )
        enable_vector_retrieval: bool = Field(
            default=True, description="🔍 启用向量检索"
        )
        enable_intelligent_chunking: bool = Field(
            default=True, description="🧩 启用智能分片"
        )
        enable_recursive_summarization: bool = Field(
            default=True, description="🔄 启用递归摘要"
        )
        enable_reranking: bool = Field(default=True, description="🔄 启用重排序")

        # ========== 智能关键字生成和上下文最大化检测 ==========
        enable_keyword_generation: bool = Field(
            default=True, description="🔑 启用智能关键字生成"
        )
        enable_ai_context_max_detection: bool = Field(
            default=True, description="🧠 启用AI上下文最大化检测"
        )
        keyword_generation_for_context_max: bool = Field(
            default=True, description="🔑 对上下文最大化启用关键字生成"
        )

        # ========== 统计和调试 ==========
        enable_detailed_stats: bool = Field(default=True, description="📊 启用详细统计")
        enable_detailed_progress: bool = Field(
            default=True, description="📱 启用详细进度显示"
        )
        debug_level: int = Field(default=0, description="🐛 调试级别 0-3")
        show_frontend_progress: bool = Field(
            default=True, description="📱 显示处理进度"
        )

        # ========== API配置 ==========
        api_error_retry_times: int = Field(default=2, description="🔄 API错误重试次数")
        api_error_retry_delay: float = Field(
            default=1.0, description="⏱️ API错误重试延迟(秒)"
        )

        # ========== Token管理 ==========
        default_token_limit: int = Field(default=200000, description="⚖️ 默认token限制")
        max_fallback_token_limit: int = Field(
            default=300000, description="🛟 未知模型最大兜底token限制"
        )
        token_safety_ratio: float = Field(
            default=0.92, description="🛡️ Token安全比例(92%)"
        )
        target_window_usage: float = Field(
            default=0.85, description="🪟 目标窗口使用率(85%)"
        )
        max_processing_iterations: int = Field(
            default=5, description="🔄 最大处理迭代次数"
        )

        # ========== 保护策略 ==========
        force_preserve_current_user_message: bool = Field(
            default=True, description="🔒 强制保留当前用户消息(最后一条用户消息)"
        )
        preserve_recent_exchanges: int = Field(
            default=4, description="💬 保护最近完整对话轮次"
        )
        max_preserve_ratio: float = Field(
            default=0.3, description="🔒 保护消息最大token比例"
        )
        max_single_message_tokens: int = Field(
            default=20000, description="📝 单条消息最大token"
        )

        # ========== 智能分片配置 ==========
        enable_smart_chunking: bool = Field(default=True, description="🧩 启用智能分片")
        chunk_target_tokens: int = Field(default=4000, description="🧩 分片目标token数")
        chunk_overlap_tokens: int = Field(default=300, description="🔗 分片重叠token数")
        chunk_min_tokens: int = Field(default=1000, description="📏 分片最小token数")
        chunk_max_tokens: int = Field(default=4000, description="📏 分片最大token数")
        large_message_threshold: int = Field(
            default=10000, description="📏 大消息分片阈值"
        )
        preserve_paragraph_integrity: bool = Field(
            default=True, description="📝 保持段落完整性"
        )
        preserve_sentence_integrity: bool = Field(
            default=True, description="📝 保持句子完整性"
        )
        preserve_code_blocks: bool = Field(
            default=True, description="💻 保持代码块完整性"
        )

        # ========== 内容优先级设置 ==========
        high_priority_content: str = Field(
            default="代码,配置,参数,数据,错误,解决方案,步骤,方法,技术细节,API,函数,类,变量,问题,bug,修复,实现,算法,架构,用户问题,关键回答",
            description="🎯 高优先级内容关键词(逗号分隔)",
        )

        # ========== 统一的API配置 ==========
        api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 API基础地址",
        )
        api_key: str = Field(default="", description="🔑 API密钥")

        # ========== 多模态模型配置 ==========
        multimodal_model: str = Field(
            default="doubao-1.5-vision-pro-250328", description="🖼️ 多模态模型"
        )

        # ========== 文本模型配置 ==========
        text_model: str = Field(
            default="doubao-1-5-lite-32k-250115", description="📝 文本处理模型"
        )

        # ========== 记忆管理模型配置 ==========
        memory_model: str = Field(
            default="doubao-1-5-lite-32k-250115", description="🧠 记忆管理模型"
        )

        # ========== 向量模型配置 ==========
        text_vector_model: str = Field(
            default="doubao-embedding-large-text-250515", description="🧠 文本向量模型"
        )
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615", description="🧠 多模态向量模型"
        )

        # ========== Vision相关配置 ==========
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。特别注意代码、配置、数据等技术信息。保持客观准确，重点突出关键信息。如果图片包含文字内容，请完整转录出来。",
            description="👁️ Vision提示词",
        )
        vision_max_tokens: int = Field(
            default=2500, description="👁️ Vision最大输出tokens"
        )

        # ========== 关键字生成配置 ==========
        keyword_generation_prompt: str = Field(
            default="""你是专业的搜索关键字生成助手。用户输入了一个查询，你需要生成多个相关的搜索关键字来帮助在对话历史中找到相关内容。
📋 任务要求：
1. 分析用户查询的意图和主题
2. 生成5-10个相关的搜索关键字
3. 包含同义词、相关词、技术术语
4. 对于宽泛查询（如"聊了什么"、"说了什么"），生成通用但有效的关键字
5. 关键字应该能覆盖可能的对话主题
📝 输出格式：
直接输出关键字，用逗号分隔，不要其他解释。
现在请为以下查询生成关键字：""",
            description="🔑 关键字生成提示词",
        )

        # ========== 上下文最大化检测配置 ==========
        context_max_detection_prompt: str = Field(
            default="""你是专业的查询意图分析助手。请分析用户的查询是否需要上下文最大化处理。
📋 判断标准：
需要上下文最大化的查询特征：
- 询问"聊了什么"、"说了什么"、"讨论了什么"等宽泛内容
- 询问"之前的内容"、"历史记录"、"对话历史"等
- 缺乏具体的主题、关键词或明确的搜索意图
- 查询词汇少于3个有效词汇
不需要上下文最大化的查询特征：
- 包含明确的主题、技术术语、产品名称等
- 有具体的问题指向
- 包含详细的描述或背景信息
📝 输出格式：
只输出 "需要上下文最大化" 或 "不需要上下文最大化"，不要其他解释。
现在请分析以下查询：""",
            description="🧠 上下文最大化检测提示词",
        )

        # ========== 向量检索配置 ==========
        vector_similarity_threshold: float = Field(
            default=0.06, description="🎯 基础相似度阈值"
        )
        multimodal_similarity_threshold: float = Field(
            default=0.04, description="🖼️ 多模态相似度阈值"
        )
        text_similarity_threshold: float = Field(
            default=0.08, description="📝 文本相似度阈值"
        )
        vector_top_k: int = Field(default=150, description="🔝 向量检索Top-K数量")

        # ========== 重排序API配置 ==========
        rerank_api_base: str = Field(
            default="https://api.bochaai.com", description="🔄 重排序API"
        )
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=100, description="🔝 重排序返回数量")

        # ========== 摘要配置 ==========
        max_summary_length: int = Field(default=25000, description="📏 摘要最大长度")
        min_summary_ratio: float = Field(
            default=0.30, description="📏 摘要最小长度比例"
        )
        summary_compression_ratio: float = Field(
            default=0.40, description="📊 摘要压缩比例"
        )
        max_recursion_depth: int = Field(default=3, description="🔄 最大递归深度")

        # ========== 性能配置 ==========
        max_concurrent_requests: int = Field(default=6, description="⚡ 最大并发数")
        request_timeout: int = Field(default=90, description="⏱️ 请求超时(秒)")

        # ========== 缓存配置 ==========
        enable_embedding_cache: bool = Field(
            default=True, description="💾 启用向量缓存"
        )
        cache_max_size: int = Field(default=1000, description="💾 缓存最大条数")

    def __init__(self):
        print("📍 高级上下文管理器 + 自动记忆 初始化中...")
        self.valves = self.Valves()

        # Auto Memory 日志
        self.logger = logging.getLogger(__name__ + ".auto_memory")

        # 初始化原有组件
        self.model_matcher = ModelMatcher()
        self.token_calculator = TokenCalculator()
        self.input_cleaner = InputCleaner()
        self.message_chunker = MessageChunker(self.token_calculator, self.valves)
        self.coverage_planner = CoveragePlanner(self.token_calculator, self.valves)

        # 初始化缓存
        if self.valves.enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(self.valves.cache_max_size)
        else:
            self.embedding_cache = None

        # 处理统计
        self.stats = ProcessingStats()

        # 消息顺序管理器
        self.message_order = None
        self.current_processing_id = None
        self.current_user_message = None
        self.current_model_info = None
        self.model_runtime_overrides: Dict[str, Dict[str, Any]] = {}
        self.forced_text_only_models: set[str] = set()
        self.image_description_cache: Dict[str, str] = {}
        self.model_multimodal_probe_results: Dict[str, bool] = {}

        # Auto Memory相关
        self.current_user_obj = None

        # 解析配置
        self._parse_configurations()
        print("✅ 初始化完成 - 高级上下文管理器 + 自动记忆")

    def _parse_configurations(self):
        """解析配置项"""
        self.high_priority_keywords = set()
        if self.valves.high_priority_content:
            self.high_priority_keywords = {
                keyword.strip().lower()
                for keyword in self.valves.high_priority_content.split(",")
                if keyword.strip()
            }

    def _normalize_model_name(self, model_name: str) -> str:
        """标准化模型名，用于运行时能力缓存"""
        return (model_name or "").strip().lower()

    def _is_model_known_non_multimodal(self, model_name: str) -> bool:
        """仅在“已知不支持多模态”时返回True，避免对未知模型过早转写"""
        model_key = self._normalize_model_name(model_name)
        if not model_key:
            return False

        if model_key in self.forced_text_only_models:
            return True

        runtime_override = self.model_runtime_overrides.get(model_key, {})
        return runtime_override.get("multimodal") is False

    def _mark_model_as_text_only(self, model_name: str, reason: str = ""):
        """在发生多模态兼容问题后，将模型标记为文本模型"""
        model_key = self._normalize_model_name(model_name)
        if not model_key:
            return

        self.forced_text_only_models.add(model_key)
        existing = self.model_runtime_overrides.get(model_key, {})
        existing["multimodal"] = False
        existing["image_tokens"] = 0
        if reason:
            existing["hint"] = reason
        self.model_runtime_overrides[model_key] = existing

    def _is_model_token_limit_known(self, model_name: str) -> bool:
        """仅在已学习到明确上下文窗口时返回True：不超限场景不预判上限。"""
        model_key = self._normalize_model_name(model_name)
        if not model_key:
            return False
        runtime_override = self.model_runtime_overrides.get(model_key, {})
        limit_val = runtime_override.get("limit")
        return isinstance(limit_val, (int, float)) and int(limit_val) > 0

    def _is_multimodal_refusal_text(self, text: str) -> bool:
        """判断文本是否在表达“无法处理图片/仅文本能力”"""
        if not text:
            return False
        lowered = str(text).lower()
        patterns = [
            r"cannot process images",
            r"can't process images",
            r"cannot view images",
            r"cannot directly display or manipulate images",
            r"limited to text",
            r"only (?:supports|support) text",
            r"i cannot process images",
            r"无法处理图片",
            r"不支持(?:图片|图像|视觉|多模态)",
            r"仅支持文本",
        ]
        return any(re.search(p, lowered, flags=re.IGNORECASE) for p in patterns)

    async def probe_model_multimodal_support(self, model_name: str) -> Optional[bool]:
        """通过极小测试图主动探测模型是否支持多模态。返回 True/False/None(未知)。"""
        model_key = self._normalize_model_name(model_name)
        if not model_key:
            return None

        cached = self.model_multimodal_probe_results.get(model_key)
        if cached is not None:
            return cached

        client = self.get_api_client()
        if not client:
            return None

        # 使用稍大测试图 + 明确握手口令，减少“图片太小/被忽略”的误判
        probe_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAK0lEQVR4nO3NQQEAAAQAMPSvIScN8DUBQGfPnj179uzZs2fPnj179uzZQw+fFQE6Y8jvGQAAAABJRU5ErkJggg=="
        probe_prompt = (
            "这是多模态能力探测。请只返回一个词："
            "如果你确实看到了图片并支持图像输入，返回 MM_OK；"
            "否则返回 MM_NO。不要输出其他内容。"
        )
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": probe_prompt},
                            {"type": "image_url", "image_url": {"url": probe_png}},
                        ],
                    }
                ],
                max_tokens=16,
                temperature=0,
                timeout=self.valves.request_timeout,
            )
        except Exception as e:
            err_text = str(e)
            await self.learn_model_capability_from_errors(
                model_name, error_text=err_text
            )

            # 探测请求本身失败（如 4xx/模型不存在/不接受图片输入）时，优先按不支持处理
            lowered_err = err_text.lower()
            probe_hard_fail = any(
                key in lowered_err
                for key in [
                    " 400",
                    " 404",
                    "not found",
                    "invalid",
                    "unsupported",
                    "does not support",
                    "image_url",
                    "vision",
                    "multimodal",
                ]
            )
            if probe_hard_fail or self._is_model_known_non_multimodal(model_name):
                self._mark_model_as_text_only(
                    model_name,
                    "多模态探测请求失败，已按文本模型兜底处理。",
                )
                self.model_multimodal_probe_results[model_key] = False
                return False
            return None

        content = ""
        if response and response.choices and response.choices[0].message:
            content = str(response.choices[0].message.content or "").lower()

        normalized = re.sub(r"\s+", "", content).upper()

        if "MM_OK" in normalized:
            self.model_multimodal_probe_results[model_key] = True
            existing = self.model_runtime_overrides.get(model_key, {})
            existing["multimodal"] = True
            if existing.get("image_tokens", 0) <= 0:
                existing["image_tokens"] = self.model_matcher.default_image_tokens
            self.model_runtime_overrides[model_key] = existing
            return True

        # 只要不是明确 MM_OK，都按不支持处理（MM_NO/拒绝文案/跑偏回复）
        self._mark_model_as_text_only(
            model_name,
            "探测未确认多模态可用（或明确拒绝图片），后续按文本模型处理。",
        )
        self.model_multimodal_probe_results[model_key] = False
        return False

    def _extract_error_signals_regex(self, text: str) -> Dict[str, Any]:
        """从错误文本中提取模型能力信号（正则兜底）"""
        if not text:
            return {}

        lowered = text.lower()
        signals: Dict[str, Any] = {}

        token_patterns = [
            r"maximum context length is\s*(\d+)",
            r"model(?:'s)? maximum context length is\s*(\d+)",
            r"max(?:imum)?(?:\s+input)?\s+tokens?\s*(?:is|are|:)\s*(\d+)",
            r"最大(?:上下文)?(?:长度|token(?:数)?)\s*(?:为|是|:)\s*(\d+)",
        ]
        for pattern in token_patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                try:
                    parsed_limit = int(match.group(1))
                    if parsed_limit > 0:
                        signals["limit"] = parsed_limit
                        break
                except Exception:
                    continue

        multimodal_unsupported_patterns = [
            r"does not support (?:image|vision|multimodal)",
            r"image(?:_url)?(?: input)? is not supported",
            r"vision is not supported",
            r"only supports text",
            r"不支持(?:图片|图像|视觉|多模态)",
            r"仅支持文本",
        ]
        if any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in multimodal_unsupported_patterns
        ):
            signals["multimodal"] = False
            signals["image_tokens"] = 0

        return signals

    def _extract_json_object(self, text: str) -> Optional[dict]:
        """从文本中提取JSON对象"""
        if not text:
            return None
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    async def _extract_error_signals_with_text_model(
        self, error_text: str
    ) -> Dict[str, Any]:
        """使用文本模型解析错误，提取模型能力信号"""
        if not error_text:
            return {}

        client = self.get_api_client()
        if not client:
            return {}

        system_prompt = (
            "你是API错误解析器。请从错误文本中识别模型能力信号，并严格输出JSON。"
            "若无法判断某字段，填 null。不要输出额外文本。"
        )
        user_prompt = (
            "错误文本：\n"
            f"{error_text}\n\n"
            "请输出："
            '{"limit": <int|null>, "multimodal": <true|false|null>, "image_tokens": <int|null>}'
        )

        try:
            response = await client.chat.completions.create(
                model=self.valves.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0,
                timeout=self.valves.request_timeout,
            )
        except Exception:
            return {}

        if not response or not response.choices:
            return {}

        content = (response.choices[0].message.content or "").strip()
        parsed = self._extract_json_object(content)
        if not parsed:
            return {}

        signals: Dict[str, Any] = {}

        limit_val = parsed.get("limit")
        if isinstance(limit_val, (int, float)) and int(limit_val) > 0:
            signals["limit"] = int(limit_val)
        elif isinstance(limit_val, str) and limit_val.strip().isdigit():
            signals["limit"] = int(limit_val.strip())

        multimodal_val = parsed.get("multimodal")
        if isinstance(multimodal_val, bool):
            signals["multimodal"] = multimodal_val

        image_tokens_val = parsed.get("image_tokens")
        if isinstance(image_tokens_val, (int, float)) and int(image_tokens_val) >= 0:
            signals["image_tokens"] = int(image_tokens_val)
        elif isinstance(image_tokens_val, str) and image_tokens_val.strip().isdigit():
            signals["image_tokens"] = int(image_tokens_val.strip())

        if signals.get("multimodal") is False and "image_tokens" not in signals:
            signals["image_tokens"] = 0

        return signals

    async def learn_model_capability_from_errors(
        self,
        model_name: str,
        error_text: str = "",
    ):
        """从请求返回错误中学习模型能力，覆盖静态字典"""
        model_key = self._normalize_model_name(model_name)
        if not model_key or not error_text:
            return

        regex_signals = self._extract_error_signals_regex(error_text)
        llm_signals = await self._extract_error_signals_with_text_model(error_text)

        merged_signals: Dict[str, Any] = {}
        merged_signals.update(regex_signals)
        merged_signals.update(llm_signals)

        if not merged_signals:
            return

        existing = self.model_runtime_overrides.get(model_key, {})
        existing.update(merged_signals)
        self.model_runtime_overrides[model_key] = existing

        if existing.get("multimodal") is False:
            self.forced_text_only_models.add(model_key)
        self.debug_log(
            1,
            f"已从错误信息学习模型能力: {model_name} -> {existing}",
            "🧠",
        )

    def reset_processing_state(self):
        """重置处理状态"""
        self.current_processing_id = None
        self.message_order = None
        self.current_user_message = None
        self.current_model_info = None
        self.stats = ProcessingStats()
        if self.embedding_cache:
            self.embedding_cache.clear()

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        """分级调试日志"""
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            message = self.input_cleaner.clean_text_for_regex(message)
            print(f"{prefix} {emoji} {message}")

    # ========== Memory相关方法 ==========

    def memory_log(self, message: str, level: str = "info"):
        """记忆系统日志（优先写入 logging，同时 print flush 便于在终端看到）"""
        if not getattr(self.valves, "enable_auto_memory", False):
            return

        prefix_map = {
            "debug": "🔍",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
        }
        prefix = prefix_map.get(level, "ℹ️")
        try:
            message = self.input_cleaner.clean_text_for_regex(message)
        except Exception:
            pass

        # 终端输出（确保 flush）
        try:
            print(f"{prefix} [AutoMemory] {message}", flush=True)
        except Exception:
            pass

        # logging 输出（给 uvicorn / docker logs）
        try:
            logger = getattr(self, "logger", None)
            if logger:
                if level == "debug":
                    logger.debug(message)
                elif level == "warning":
                    logger.warning(message)
                elif level == "error":
                    logger.error(message)
                else:
                    logger.info(message)
        except Exception:
            pass

    def messages_to_string_for_memory(self, messages: list[dict]) -> str:
        """将消息转换为字符串格式供记忆系统使用"""
        stringified_messages = []
        effective_count = self.valves.memory_messages_to_consider

        for i in range(1, effective_count + 1):
            if i > len(messages):
                break
            try:
                message = messages[-i]
                stringified_messages.append(
                    STRINGIFIED_MESSAGE_TEMPLATE.format(
                        index=i,
                        role=message.get("role", "user"),
                        content=self.extract_text_from_content(
                            message.get("content", "")
                        ),
                    )
                )
            except Exception as e:
                self.memory_log(f"消息字符串化失败 {i}: {e}", "warning")

        return "\n".join(stringified_messages)

    async def get_related_memories_for_auto_memory(
        self, messages: list[dict], user: UserModel
    ) -> list[Memory]:
        """获取相关记忆"""
        memory_query = self.build_memory_query_from_messages(messages)

        try:
            results = await query_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=QueryMemoryForm(
                    content=memory_query, k=self.valves.memory_related_memories_n
                ),
                user=user,
            )
        except HTTPException as e:
            if e.status_code == 404:
                # Open WebUI 常见行为：当用户尚无任何记忆时，会返回 404（例如 detail 为 "No memories found for user"）。
                # 这不代表 Memory 功能不可用；仅表示当前没有可检索的记忆。
                self.memory_log(f"未找到相关记忆（404）: {e.detail}", "info")
                return []
            else:
                self.memory_log(f"记忆查询失败 {e.status_code}: {e.detail}", "error")
                raise RuntimeError("记忆查询失败") from e
        except Exception as e:
            self.memory_log(f"记忆查询异常: {e}", "error")
            raise RuntimeError("记忆查询失败") from e

        related_memories = searchresults_to_memories(results) if results else []
        self.memory_log(f"找到 {len(related_memories)} 条相关记忆", "info")

        if self.valves.memory_minimum_similarity is not None:
            filtered_memories = [
                mem
                for mem in related_memories
                if mem.similarity_score is not None
                and mem.similarity_score >= self.valves.memory_minimum_similarity
            ]
            filtered_count = len(related_memories) - len(filtered_memories)
            if filtered_count > 0:
                self.memory_log(f"过滤掉 {filtered_count} 条低相似度记忆", "info")
            related_memories = filtered_memories

        return related_memories

    def build_memory_query_from_messages(self, messages: list[dict]) -> str:
        """从消息构建记忆查询"""
        query_parts = []

        last_user_idx = None
        last_user_msg = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                last_user_msg = messages[idx].get("content", "")
                break

        if last_user_msg is None or last_user_idx is None:
            return ""

        user_text = self.extract_text_from_content(last_user_msg)
        user_word_count = len(user_text.split())
        include_context = user_word_count <= 8

        if last_user_idx + 1 < len(messages):
            assistant_msg = self.extract_text_from_content(
                messages[last_user_idx + 1].get("content", "")
            )
            if assistant_msg:
                query_parts.append(f"Assistant: {assistant_msg}")

        query_parts.append(f"User: {user_text}")

        if include_context and last_user_idx > 0:
            prev_msg = self.extract_text_from_content(
                messages[last_user_idx - 1].get("content", "")
            )
            if prev_msg and messages[last_user_idx - 1].get("role") == "assistant":
                query_parts.append(f"Assistant: {prev_msg}")

        query_parts.reverse()
        return "\n".join(query_parts)

    async def query_memory_llm_for_actions(
        self,
        conversation_str: str,
        stringified_memories: str,
        existing_ids: list[str],
        event_emitter,
    ):
        """调用 LLM 获取记忆操作（兼容 openai-compatible：不依赖 .parse）"""
        client = self.get_api_client()
        if not client:
            self.memory_log(
                "API客户端初始化失败（请检查 api_base/api_key 配置）", "error"
            )
            return None

        model_to_use = self.valves.memory_model or self.valves.model
        ids_hint = ", ".join(existing_ids) if existing_ids else "(none)"
        output_rules = (
            AUTO_MEMORY_OUTPUT_INSTRUCTIONS
            + "\nExisting IDs for update/delete:\n"
            + ids_hint
            + "\n"
        )

        messages = [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    'Conversation snippet (plain text, latest user message is the LAST line that starts with "User:"):\n'
                    f"{conversation_str}\n\n"
                    "Related Memories (may be empty):\n"
                    f"{stringified_memories}\n\n"
                    f"{output_rules}"
                ),
            },
        ]

        # 统一走 create；某些环境 openai==1.x 可能没有 chat.completions.parse
        try:
            response = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.3,
                timeout=self.valves.request_timeout,
            )
        except TypeError:
            # 某些实现不支持 timeout 参数
            response = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.3,
            )
        except Exception as e:
            self.memory_log(f"LLM调用失败: {str(e)[:200]}", "error")
            return None

        try:
            if not response.choices:
                self.memory_log("LLM返回空 choices", "error")
                return None

            text_response = (response.choices[0].message.content or "").strip()
            if not text_response:
                self.memory_log("LLM返回空响应", "error")
                return None

            # 去掉可能的 code fence
            if text_response.startswith("```"):
                text_response = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text_response)
                text_response = re.sub(r"\n```\s*$", "", text_response).strip()

            response_model = build_actions_request_model(existing_ids)

            # 尝试提取 JSON 对象（兼容前后夹杂解释文字的情况）
            json_str = text_response
            l = json_str.find("{")
            r = json_str.rfind("}")
            if l != -1 and r != -1 and r > l:
                json_str = json_str[l : r + 1]
            else:
                # 常见：模型直接返回“no actions needed ...”等非 JSON 文本
                if re.search(
                    r"\bno actions?\b|\bno action needed\b", json_str, re.IGNORECASE
                ):
                    self.memory_log(
                        "LLM返回非JSON的无操作结果，视为 actions=[]", "info"
                    )
                    return response_model.model_validate(
                        {
                            "actions": [],
                            "reason": "LLM returned non-JSON no-action result",
                        }
                    )
                if not json_str.strip():
                    self.memory_log("LLM返回空结果，视为 actions=[]", "info")
                    return response_model.model_validate(
                        {
                            "actions": [],
                            "reason": "LLM returned non-JSON no-action result",
                        }
                    )

            try:
                return response_model.model_validate_json(json_str)
            except ValidationError:
                # 有些模型会返回单引号/尾逗号等，降级：先 json.loads 再 validate
                obj = json.loads(json_str)
                return response_model.model_validate(obj)

        except Exception as e:
            preview = (locals().get("text_response", "") or "")[:400]
            self.memory_log(f"记忆操作解析失败: {e}; raw={preview!r}", "error")
            return None

    async def apply_memory_actions(
        self,
        action_plan: MemoryActionRequestStub,
        user: UserModel,
        emitter: Callable,
    ):
        """执行记忆操作（即使 0 actions 也会打印日志，便于排障）"""
        if not action_plan:
            self.memory_log("action_plan 为空，跳过记忆操作", "warning")
            return

        actions = list(getattr(action_plan, "actions", None) or [])
        self.memory_log(f"收到 {len(actions)} 个记忆操作", "info")

        reason = (getattr(action_plan, "reason", "") or "").strip()
        if not actions and reason:
            # 让 actions=0 的情况更直观
            self.memory_log(f"无记忆更新原因: {reason}", "info")

        if not actions:
            # 0 actions 的情况很常见（LLM 判断无需写入/更新/删除记忆）

            self.memory_log("无记忆更新（actions=0）", "info")
            return

        self.memory_log(f"开始执行 {len(actions)} 个记忆操作", "info")

        operations = {
            "delete": {
                "actions": [a for a in actions if a.action == "delete"],
                "handler": lambda a: delete_memory_by_id(memory_id=a.id, user=user),
                "status_verb": "deleted",
            },
            "update": {
                "actions": [a for a in actions if a.action == "update"],
                "handler": lambda a: update_memory_by_id(
                    memory_id=a.id,
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=MemoryUpdateModel(content=a.content),
                    user=user,
                ),
                "status_verb": "updated",
            },
            "add": {
                "actions": [a for a in actions if a.action == "add"],
                "handler": lambda a: add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=a.content),
                    user=user,
                ),
                "status_verb": "saved",
            },
        }

        counts = {}
        for op_name, op_config in operations.items():
            counts[op_name] = 0
            for action in op_config["actions"]:
                if op_name in ["add", "update"]:
                    content = getattr(action, "content", "")
                    if not content or not content.strip():
                        continue

                try:
                    await op_config["handler"](action)
                    counts[op_name] += 1
                    self.memory_log(
                        f"{op_config['status_verb']}: {getattr(action, 'id', 'new')}",
                        "info",
                    )
                except Exception as e:
                    # 兼容不同 Open WebUI 版本：部分版本的 router 会因为“Memory 开关/权限/路由差异”等返回 404/403
                    # 这里做 DB 层 fallback（与社区 Memory 工具同一路径：open_webui.models.memories.Memories）
                    self.memory_log(
                        f"操作失败 ({op_name})，准备尝试 fallback: {str(e)[:120]}",
                        "warning",
                    )

                    try:
                        from open_webui.models.memories import Memories  # type: ignore

                        if op_name == "add":
                            content = getattr(action, "content", "")
                            fb = Memories.insert_new_memory(user.id, content)
                            if fb:
                                counts[op_name] += 1
                                self.memory_log(
                                    f"fallback saved: {getattr(fb, 'id', 'new')}",
                                    "info",
                                )
                                continue
                        elif op_name == "update":
                            content = getattr(action, "content", "")
                            fb = Memories.update_memory_by_id(action.id, content)
                            if fb:
                                counts[op_name] += 1
                                self.memory_log(
                                    f"fallback updated: {action.id}", "info"
                                )
                                continue
                        elif op_name == "delete":
                            fb = Memories.delete_memory_by_id(action.id)
                            if fb:
                                counts[op_name] += 1
                                self.memory_log(
                                    f"fallback deleted: {action.id}", "info"
                                )
                                continue
                    except Exception as fb_e:
                        self.memory_log(
                            f"fallback 失败 ({op_name}): {str(fb_e)[:120]}", "error"
                        )

                    self.memory_log(
                        f"操作失败 ({op_name}) 且 fallback 失败: {str(e)[:120]}",
                        "error",
                    )

        status_parts = []
        for op_name, op_config in operations.items():
            count = counts[op_name]
            if count > 0:
                memory_word = "memory" if count == 1 else "memories"
                status_parts.append(f"{op_config['status_verb']} {count} {memory_word}")

        status_message = ", ".join(status_parts)
        if status_message:
            self.memory_log(f"记忆操作结果: {status_message}", "info")

        # 额外：打印 DB 中的记忆条数，方便判断“到底有没有写进去”
        try:
            from open_webui.models.memories import Memories  # type: ignore

            _all = Memories.get_memories_by_user_id(user.id) or []
            self.memory_log(f"当前数据库记忆条数: {len(_all)}", "info")
            if _all:
                _latest = sorted(_all, key=lambda m: getattr(m, "created_at", 0))[-1]
                _preview = (getattr(_latest, "content", "") or "")[:80].replace(
                    "\n", " "
                )
                self.memory_log(
                    f"最新记忆预览: {getattr(_latest, 'id', '')} :: {_preview!r}",
                    "debug",
                )
        except Exception as e:
            self.memory_log(
                f"读取数据库记忆失败(仅用于日志): {str(e)[:120]}", "warning"
            )

        self.memory_log(status_message or "无记忆更新", "info")

    async def auto_memory_process(
        self,
        messages: list[dict],
        user: UserModel,
        emitter: Callable,
    ):
        """自动记忆处理主流程"""
        if len(messages) < 2:
            self.memory_log("消息数不足，跳过记忆处理", "debug")
            return

        self.memory_log(f"开始记忆处理 - 用户: {user.id}", "info")

        try:
            # 1) 可选：强制写入记忆（用于调试/显式指令），命中则跳过LLM判断
            prefixes_raw = (
                getattr(self.valves, "memory_force_add_prefixes", "") or ""
            ).strip()
            if prefixes_raw:
                prefixes = [p.strip() for p in prefixes_raw.split(";") if p.strip()]
                latest_user = next(
                    (m for m in reversed(messages) if m.get("role") == "user"), None
                )
                latest_content = (latest_user or {}).get("content", "")
                # 将 content 统一转为纯文本
                if isinstance(latest_content, list):
                    text_parts = []
                    for part in latest_content:
                        if isinstance(part, dict):
                            if part.get("type") == "text" and part.get("text"):
                                text_parts.append(str(part.get("text")))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    latest_text = "\n".join([t for t in text_parts if t]).strip()
                else:
                    latest_text = str(latest_content or "").strip()

                for pfx in prefixes:
                    if latest_text.lower().startswith(pfx.lower()):
                        forced_text = latest_text[len(pfx) :].strip()
                        if forced_text:
                            self.memory_log(
                                f"检测到强制记忆前缀，直接写入: {forced_text[:120]}",
                                "info",
                            )
                            forced_plan = MemoryActionRequestStub(
                                actions=[
                                    MemoryAddAction(action="add", content=forced_text)
                                ],
                                reason=f"forced_prefix:{pfx}",
                            )
                            await self.apply_memory_actions(forced_plan, user, emitter)
                            self.memory_log("记忆处理完成（强制写入）", "info")
                            return

            # 2) 正常：检索相关记忆 + 让LLM决定写入/更新/删除
            related_memories = await self.get_related_memories_for_auto_memory(
                messages, user
            )
            stringified_memories = json.dumps(
                [memory.model_dump(mode="json") for memory in related_memories]
            )
            conversation_str = self.messages_to_string_for_memory(messages)
            existing_ids = [m.mem_id for m in related_memories]

            action_plan = await self.query_memory_llm_for_actions(
                conversation_str, stringified_memories, existing_ids, emitter
            )

            if not action_plan:
                self.memory_log("LLM未返回有效操作", "warning")
                return

            await self.apply_memory_actions(action_plan, user, emitter)
            self.memory_log("记忆处理完成", "info")

        except Exception as e:
            self.memory_log(f"记忆处理异常: {str(e)[:200]}", "error")
            import traceback

            if self.valves.debug_level >= 2:
                traceback.print_exc()

    def extract_memory_context(self, content: str) -> Optional[tuple[str, list[dict]]]:
        """从系统消息中提取记忆上下文"""
        pattern = r"<memory_user_context>\s*(\[[\s\S]*?\])\s*</memory_user_context>"
        match = re.search(pattern, content)
        if not match:
            return None

        try:
            memories_json = match.group(1)
            memories_list = json.loads(memories_json)
            self.memory_log(f"提取到 {len(memories_list)} 条记忆", "debug")
            return (match.group(0), memories_list)
        except json.JSONDecodeError as e:
            self.memory_log(f"记忆上下文JSON解析失败: {e}", "error")
            return None

    def format_memory_context(self, memories: list[dict]) -> str:
        """格式化记忆上下文"""
        memories = [
            {k: v for k, v in mem.items() if k != "similarity_score"}
            for mem in memories
        ]
        memories_json = json.dumps(memories, indent=2, ensure_ascii=False)
        return f"<long_term_memory>\n{memories_json}\n</long_term_memory>"

    def process_memory_context_in_messages(self, messages: list[dict]) -> list[dict]:
        """处理消息中的记忆上下文"""
        if not self.valves.override_memory_context:
            return messages

        found_any = False
        for i, message in enumerate(messages):
            if message.get("role") != "system":
                continue

            content = message.get("content", "")
            if not content:
                continue

            extraction_result = self.extract_memory_context(content)
            if extraction_result:
                found_any = True
                full_match, memories_list = extraction_result
                new_context = self.format_memory_context(memories_list)
                messages[i]["content"] = content.replace(full_match, new_context)
                self.memory_log(
                    f"覆盖系统消息{i}的记忆上下文: {len(memories_list)}条记忆", "info"
                )

        if not found_any and self.valves.override_memory_context:
            self.memory_log("未找到记忆上下文标签", "warning")

        return messages

    # ========== 工具方法 ==========

    def is_model_excluded(self, model_name: str) -> bool:
        """检查模型是否被排除"""
        if not self.valves.excluded_models or not model_name:
            return False
        excluded_list = [
            model.strip().lower()
            for model in self.valves.excluded_models.split(",")
            if model.strip()
        ]
        if not excluded_list:
            return False
        model_lower = model_name.lower()
        for excluded_model in excluded_list:
            if excluded_model in model_lower:
                self.debug_log(1, f"模型 {model_name} 在排除列表中", "🚫")
                return True
        return False

    def analyze_model(self, model_name: str) -> Dict[str, Any]:
        """分析模型信息"""
        model_info = self.model_matcher.match_model(model_name)

        model_key = self._normalize_model_name(model_name)
        runtime_override = self.model_runtime_overrides.get(model_key)
        if runtime_override:
            model_info.update(runtime_override)
            model_info["match_type"] = "runtime"

        self.token_calculator.set_model_info(model_info)

        multimodal_status = "多模态" if model_info["multimodal"] else "文本"
        family_name = model_info["family"].upper()
        tokens_display = f"{model_info['limit']:,}tokens"
        match_type = model_info.get("match_type")
        if match_type == "exact":
            match_type_display = "精确"
        elif match_type == "fuzzy":
            match_type_display = "模糊"
        elif match_type == "runtime":
            match_type_display = "错误学习"
        else:
            match_type_display = "默认"

        print(f"🎯 模型识别: {model_name}")
        print(f"   ├─ 系列: {family_name}")
        print(f"   ├─ 类型: {multimodal_status}")
        print(f"   ├─ 限制: {tokens_display}")
        print(f"   └─ 匹配: {match_type_display}匹配")
        if model_info.get("hint"):
            print(f"   ⚠️ 提示: {model_info['hint']}")

        if model_info.get("special") == "thinking":
            print(f"   💭 特殊: Thinking模型")

        if model_info.get("family") == "gpt" and "gpt-5" in model_name.lower():
            print(f"   🆕 新模型: GPT-5系列 (200k tokens + 多模态)")

        return model_info

    def count_tokens(self, text: str) -> int:
        """简化的token计算"""
        if not text:
            return 0
        return self.token_calculator.count_tokens(text)

    def count_message_tokens(self, message: dict) -> int:
        """计算单条消息的token数量"""
        if not message:
            return 0
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
                elif item.get("type") == "image_url":
                    total_tokens += self.token_calculator.calculate_image_tokens("")
        else:
            total_tokens = self.count_tokens(content)

        total_tokens += self.count_tokens(role) + 20
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数量"""
        if not messages:
            return 0
        total_tokens = sum(self.count_message_tokens(msg) for msg in messages)
        self.debug_log(
            2,
            f"消息列表token计算: {len(messages)}条消息 -> {total_tokens:,}tokens",
            "📊",
        )
        return total_tokens

    def strip_internal_fields(self, messages: List[dict]) -> List[dict]:
        """移除消息中的内部字段"""
        if not messages:
            return []

        cleaned_messages: List[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned_messages.append(msg)
                continue

            new_msg: dict = {k: v for k, v in msg.items() if not str(k).startswith("_")}
            content = new_msg.get("content")

            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                        new_item = {
                            k: v for k, v in item.items() if not str(k).startswith("_")
                        }
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content

            cleaned_messages.append(new_msg)

        return cleaned_messages

    def get_model_token_limit(self, model_name: str) -> int:
        """获取模型的token限制"""
        model_info = self.analyze_model(model_name)

        # 未学习到真实上限前，使用“最大兜底限制”，避免过早触发压缩
        if not self._is_model_token_limit_known(model_name):
            limit = int(
                getattr(self.valves, "max_fallback_token_limit", 300000) or 300000
            )
            source = "fallback"
        else:
            limit = int(model_info.get("limit", self.valves.default_token_limit))
            source = "learned"

        safe_limit = int(limit * self.valves.token_safety_ratio)
        self.debug_log(
            2,
            f"模型token限制[{source}]: {model_name} -> {limit} -> {safe_limit}",
            "⚖️",
        )
        return safe_limit

    def is_multimodal_model(self, model_name: str) -> bool:
        """判断模型是否支持多模态输入"""
        model_info = self.analyze_model(model_name)
        return model_info.get("multimodal", False)

    def find_current_user_message(self, messages: List[dict]) -> Optional[dict]:
        """查找当前用户消息"""
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                self.debug_log(
                    2,
                    f"找到当前用户消息: {len(self.extract_text_from_content(msg.get('content', '')))}字符",
                    "💬",
                )
                return msg
        return None

    def separate_current_and_history_messages(
        self, messages: List[dict]
    ) -> Tuple[Optional[dict], List[dict]]:
        """分离当前用户消息和历史消息"""
        if not messages:
            return None, []

        current_user_message = None
        current_user_index = -1

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                current_user_message = msg
                current_user_index = i
                break

        if not current_user_message:
            self.debug_log(1, "未找到当前用户消息，所有消息作为历史消息处理", "⚠️")
            return None, messages

        history_messages = messages[:current_user_index]
        self.stats.history_message_separation_count += 1
        self.debug_log(
            1,
            f"消息分离完成: 当前用户消息1条({self.count_message_tokens(current_user_message)}tokens), 历史消息{len(history_messages)}条({self.count_messages_tokens(history_messages):,}tokens)",
            "📋",
        )
        return current_user_message, history_messages

    def calculate_target_tokens(self, model_name: str, current_user_tokens: int) -> int:
        """计算目标token数"""
        model_token_limit = self.get_model_token_limit(model_name)
        response_buffer = min(
            self.valves.response_buffer_max,
            max(
                self.valves.response_buffer_min,
                int(model_token_limit * self.valves.response_buffer_ratio),
            ),
        )
        target_tokens = model_token_limit - current_user_tokens - response_buffer
        min_target = max(10000, model_token_limit * 0.3)
        target_tokens = max(target_tokens, min_target)
        self.debug_log(
            1,
            f"目标token计算: {model_token_limit} - {current_user_tokens} - {response_buffer} = {target_tokens}",
            "🎯",
        )
        return int(target_tokens)

    def _needs_processing(
        self, messages: List[dict], model_name: str, target_tokens: int
    ):
        """判定是否需要进行处理"""
        current_tokens = self.count_messages_tokens(messages)
        has_images = self.has_images_in_messages(messages)
        token_limit_known = self._is_model_token_limit_known(model_name)
        token_overflow = token_limit_known and (current_tokens > target_tokens)
        # 多模态不兼容仅在“已知不支持”时触发，未知模型先直通
        multimodal_incompatible = has_images and self._is_model_known_non_multimodal(
            model_name
        )
        return (
            (token_overflow or multimodal_incompatible),
            token_overflow,
            multimodal_incompatible,
        )

    def should_force_maximize_content(
        self, messages: List[dict], target_tokens: int
    ) -> bool:
        """判断是否应该强制进行内容最大化处理"""
        current_tokens = self.count_messages_tokens(messages)
        return current_tokens > target_tokens

    # ========== 多模态处理 ==========

    def has_images_in_content(self, content) -> bool:
        """检查内容中是否包含图片"""
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return False

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        """检查消息列表中是否包含图片"""
        return any(self.has_images_in_content(msg.get("content")) for msg in messages)

    def extract_text_from_content(self, content) -> str:
        """从内容中提取文本"""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    text_parts.append(text)
            return " ".join(text_parts)
        else:
            return str(content) if content else ""

    def extract_images_from_content(self, content) -> List[dict]:
        """从内容中提取图片信息"""
        if isinstance(content, list):
            images = []
            for item in content:
                if item.get("type") == "image_url":
                    images.append(item)
            return images
        return []

    def is_high_priority_content(self, text: str) -> bool:
        """判断是否为高优先级内容"""
        if not text or not self.high_priority_keywords:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.high_priority_keywords)

    # ========== API客户端管理 ==========

    def get_api_client(self, client_type: str = "default"):
        """获取API客户端"""
        if not OPENAI_AVAILABLE:
            return None
        if self.valves.api_key:
            return AsyncOpenAI(
                base_url=self.valves.api_base,
                api_key=self.valves.api_key,
                timeout=self.valves.request_timeout,
            )
        return None

    # ========== 安全API调用 ==========

    async def safe_api_call(self, call_func, call_name: str, *args, **kwargs):
        """安全的API调用包装器"""
        for attempt in range(self.valves.api_error_retry_times + 1):
            try:
                result = await call_func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                self.stats.api_failures += 1

                # 失败学习：从报错中学习模型能力（token上限/多模态支持）
                try:
                    fallback_model = getattr(self, "_current_model_name", "")
                    await self.learn_model_capability_from_errors(
                        fallback_model,
                        error_text=error_msg,
                    )
                except Exception:
                    pass

                if attempt < self.valves.api_error_retry_times:
                    self.debug_log(
                        1,
                        f"{call_name} 第{attempt+1}次尝试失败，{self.valves.api_error_retry_delay}秒后重试",
                        "🔄",
                    )
                    await asyncio.sleep(self.valves.api_error_retry_delay)
                else:
                    self.debug_log(1, f"{call_name} 最终失败: {error_msg[:100]}", "❌")
                    return None
        return None

    # ========== 上下文最大化检测 ==========

    async def detect_context_max_need_impl(self, query_text: str, event_emitter):
        """实际的上下文最大化检测实现"""
        client = self.get_api_client()
        if not client:
            return None

        cleaned_query = self.input_cleaner.clean_text_for_regex(query_text)
        prompt = f"{self.valves.context_max_detection_prompt}\n\n{cleaned_query}"

        response = await client.chat.completions.create(
            model=self.valves.text_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
            timeout=self.valves.request_timeout,
        )

        if response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content.strip()
            result = self.input_cleaner.clean_text_for_regex(result)
            need_context_max = "需要上下文最大化" in result
            self.debug_log(
                2, f"AI上下文最大化检测结果: {result} -> {need_context_max}", "🧠"
            )
            return need_context_max
        return None

    async def detect_context_max_need(self, query_text: str, event_emitter) -> bool:
        """使用AI检测是否需要上下文最大化"""
        if not self.valves.enable_ai_context_max_detection:
            return self.is_context_max_need_simple(query_text)

        self.debug_log(1, f"AI检测上下文最大化需求: {query_text[:50]}...", "🧠")
        need_context_max = await self.safe_api_call(
            self.detect_context_max_need_impl,
            "上下文最大化检测",
            query_text,
            event_emitter,
        )

        if need_context_max is not None:
            self.stats.context_maximization_detections += 1
            self.debug_log(
                1,
                f"AI上下文最大化检测完成: {'需要' if need_context_max else '不需要'}",
                "🧠",
            )
            return need_context_max
        else:
            self.debug_log(1, f"AI检测失败，使用简单方法", "⚠️")
            return self.is_context_max_need_simple(query_text)

    def is_context_max_need_simple(self, query_text: str) -> bool:
        """简单的上下文最大化需求判断"""
        if not query_text:
            return True

        query_text = self.input_cleaner.clean_text_for_regex(query_text)
        context_max_patterns = [
            r".*聊.*什么.*",
            r".*说.*什么.*",
            r".*讨论.*什么.*",
            r".*谈.*什么.*",
            r".*内容.*",
            r".*话题.*",
            r".*历史.*",
            r".*记录.*",
            r".*之前.*",
            r"what.*discuss.*",
            r"what.*talk.*",
            r"what.*chat.*",
            r".*conversation.*",
            r".*history.*",
        ]

        query_lower = query_text.lower()
        for pattern in context_max_patterns:
            if self.input_cleaner.safe_regex_match(pattern, query_lower):
                return True

        return len(query_text.split()) <= 3

    # ========== 关键字生成 ==========

    async def generate_keywords_impl(self, query_text: str, event_emitter):
        """实际的关键字生成实现"""
        client = self.get_api_client()
        if not client:
            return None

        cleaned_query = self.input_cleaner.clean_text_for_regex(query_text)
        prompt = f"{self.valves.keyword_generation_prompt}\n\n{cleaned_query}"

        response = await client.chat.completions.create(
            model=self.valves.text_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            timeout=self.valves.request_timeout,
        )

        if response.choices and response.choices[0].message.content:
            keywords_text = response.choices[0].message.content.strip()
            keywords_text = self.input_cleaner.clean_text_for_regex(keywords_text)
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            keywords = [kw for kw in keywords if len(kw) >= 2]
            self.debug_log(2, f"生成关键字: {keywords[:5]}...", "🔑")
            return keywords
        return None

    async def generate_search_keywords(
        self, query_text: str, event_emitter
    ) -> List[str]:
        """生成搜索关键字"""
        if not self.valves.enable_keyword_generation:
            return [query_text]

        need_context_max = await self.detect_context_max_need(query_text, event_emitter)

        if not need_context_max and not self.valves.keyword_generation_for_context_max:
            self.debug_log(2, f"具体查询，使用原始文本: {query_text[:50]}...", "🔑")
            return [query_text]

        self.debug_log(1, f"生成搜索关键字: {query_text[:50]}...", "🔑")
        keywords = await self.safe_api_call(
            self.generate_keywords_impl,
            "关键字生成",
            query_text,
            event_emitter,
        )

        if keywords:
            final_keywords = [query_text] + keywords
            final_keywords = list(dict.fromkeys(final_keywords))
            self.stats.keyword_generations += 1
            self.debug_log(1, f"关键字生成完成: {len(final_keywords)}个", "🔑")
            return final_keywords
        else:
            self.debug_log(1, f"关键字生成失败，使用原始查询", "⚠️")
            return [query_text]

    # ========== 向量处理 ==========

    async def get_text_embedding_impl(self, text: str, event_emitter):
        """实际的文本向量获取实现"""
        client = self.get_api_client()
        if not client:
            return None

        cleaned_text = self.input_cleaner.clean_text_for_regex(text)
        self.stats.embedding_requests += 1

        response = await client.embeddings.create(
            model=self.valves.text_vector_model,
            input=[cleaned_text[:8000]],
            encoding_format="float",
        )

        if (
            response
            and response.data
            and len(response.data) > 0
            and response.data[0].embedding
        ):
            return response.data[0].embedding
        return None

    async def get_text_embedding(
        self, text: str, event_emitter
    ) -> Optional[List[float]]:
        """获取文本向量 - 带缓存"""
        if not text:
            return None

        content_key = hashlib.md5(text.encode()).hexdigest()[:16]

        if self.embedding_cache:
            cached_embedding = self.embedding_cache.get(content_key)
            if cached_embedding:
                self.stats.cache_hits += 1
                self.debug_log(3, f"文本向量缓存命中: {len(cached_embedding)}维", "💾")
                return cached_embedding

        self.stats.cache_misses += 1
        embedding = await self.safe_api_call(
            self.get_text_embedding_impl,
            "文本向量",
            text,
            event_emitter,
        )

        if embedding:
            if self.embedding_cache:
                self.embedding_cache.set(content_key, embedding)
            self.debug_log(3, f"文本向量获取成功: {len(embedding)}维", "📝")

        return embedding

    async def get_multimodal_embedding_impl(self, content, event_emitter):
        """实际的多模态向量获取实现"""
        client = self.get_api_client()
        if not client:
            return None

        if isinstance(content, list):
            cleaned_content = []
            for item in content:
                if item.get("type") == "text":
                    cleaned_item = item.copy()
                    text = item.get("text", "")
                    cleaned_text = self.input_cleaner.clean_text_for_regex(text)
                    cleaned_item["text"] = cleaned_text
                    cleaned_content.append(cleaned_item)
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    is_valid, cleaned_url = (
                        self.input_cleaner.validate_and_clean_image_url(image_url)
                    )
                    if is_valid:
                        cleaned_item = copy.deepcopy(item)
                        cleaned_item["image_url"]["url"] = cleaned_url
                        cleaned_content.append(cleaned_item)
                else:
                    cleaned_content.append(item)
            input_data = cleaned_content
        else:
            text = str(content)
            cleaned_text = self.input_cleaner.clean_text_for_regex(text)
            input_data = [{"type": "text", "text": cleaned_text[:8000]}]

        self.stats.embedding_requests += 1

        try:
            response = await client.embeddings.create(
                model=self.valves.multimodal_vector_model,
                input=input_data,
                encoding_format="float",
            )

            if hasattr(response, "data") and hasattr(response.data, "embedding"):
                return response.data.embedding
            elif (
                hasattr(response, "data")
                and isinstance(response.data, list)
                and len(response.data) > 0
            ):
                return response.data[0].embedding
            else:
                self.debug_log(1, f"多模态向量响应格式异常", "⚠️")
                return None
        except Exception as e:
            self.debug_log(1, f"多模态向量调用失败: {str(e)[:100]}", "❌")
            raise

    async def get_multimodal_embedding(
        self, content, event_emitter
    ) -> Optional[List[float]]:
        """获取多模态向量"""
        if not content:
            return None

        has_multimodal_content = False
        if isinstance(content, list):
            has_multimodal_content = any(
                item.get("type") in ["image_url", "video_url"] for item in content
            )

        if not has_multimodal_content:
            self.debug_log(3, "内容不包含多模态元素，不使用多模态向量", "📝")
            return None

        embedding = await self.safe_api_call(
            self.get_multimodal_embedding_impl,
            "多模态向量",
            content,
            event_emitter,
        )

        if embedding:
            self.debug_log(3, f"多模态向量获取成功: {len(embedding)}维", "🖼️")

        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    # ========== 相关度计算 ==========

    async def compute_relevance_scores(
        self, query_msg: dict, history_msgs: List[dict], progress: ProgressTracker
    ) -> List[dict]:
        """计算所有历史消息的相关度分数"""
        if not history_msgs:
            return []

        self.debug_log(
            1, f"开始计算相关度分数: 查询1条，历史{len(history_msgs)}条", "🎯"
        )

        query_content = query_msg.get("content", "")
        query_text = self.extract_text_from_content(query_content)

        if len(history_msgs) > 40:
            lightweight_scored = self._compute_lightweight_scores(
                query_text, history_msgs
            )
            top_k = min(self.valves.vector_top_k, len(lightweight_scored))
            lightweight_scored.sort(key=lambda x: x["score"], reverse=True)
            selected_msgs = lightweight_scored[:top_k]
            self.debug_log(
                1,
                f"两阶段召回: {len(history_msgs)} -> {len(selected_msgs)}条进入向量化阶段",
                "⚡",
            )

            if len(selected_msgs) > 80:
                scored = selected_msgs
            else:
                if self.has_images_in_content(query_content):
                    query_vector = await self.get_multimodal_embedding(
                        query_content, progress.event_emitter
                    )
                    if not query_vector:
                        query_vector = await self.get_text_embedding(
                            query_text, progress.event_emitter
                        )
                else:
                    query_vector = await self.get_text_embedding(
                        query_text, progress.event_emitter
                    )

                scored = await self._compute_vector_scores_concurrent(
                    query_vector, selected_msgs, progress
                )
        else:
            if self.has_images_in_content(query_content):
                query_vector = await self.get_multimodal_embedding(
                    query_content, progress.event_emitter
                )
                if not query_vector:
                    query_vector = await self.get_text_embedding(
                        query_text, progress.event_emitter
                    )
            else:
                query_vector = await self.get_text_embedding(
                    query_text, progress.event_emitter
                )

            msg_items = []
            for idx, msg in enumerate(history_msgs):
                msg_items.append(
                    {"msg": msg, "idx": idx, "tokens": self.count_message_tokens(msg)}
                )

            scored = await self._compute_vector_scores_concurrent(
                query_vector, msg_items, progress
            )

        self.debug_log(1, f"相关度计算完成: {len(scored)}条消息全部评分", "🎯")

        if self.valves.debug_level >= 2:
            top5 = sorted(scored, key=lambda x: x["score"], reverse=True)[:5]
            for i, item in enumerate(top5):
                self.debug_log(
                    2,
                    f"Top{i+1}: score={item['score']:.3f}, {item['tokens']}tokens",
                    "📊",
                )

        return scored

    def _compute_lightweight_scores(
        self, query_text: str, history_msgs: List[dict]
    ) -> List[dict]:
        """轻量级评分"""
        scored = []
        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        for idx, msg in enumerate(history_msgs):
            msg_content = msg.get("content", "")
            msg_text = self.extract_text_from_content(msg_content)
            msg_lower = msg_text.lower()
            msg_words = set(msg_lower.split())

            common_words = query_words & msg_words
            text_sim = (
                len(common_words) / max(1, len(query_words)) if query_words else 0
            )

            recency = idx / max(1, len(history_msgs) - 1)
            role = msg.get("role", "")
            role_weight = (
                1.0 if role == "user" else (0.8 if role == "assistant" else 0.6)
            )
            kw_bonus = 1.0 if self.is_high_priority_content(msg_text) else 0.0

            score = 0.6 * text_sim + 0.2 * recency + 0.1 * role_weight + 0.1 * kw_bonus

            scored.append(
                {
                    "msg": msg,
                    "score": score,
                    "tokens": self.count_message_tokens(msg),
                    "idx": idx,
                    "sim": text_sim,
                    "recency": recency,
                    "role_weight": role_weight,
                    "kw_bonus": kw_bonus,
                }
            )

        return scored

    async def _compute_vector_scores_concurrent(
        self,
        query_vector: List[float],
        msg_items: List[dict],
        progress: ProgressTracker,
    ) -> List[dict]:
        """并发计算向量分数"""
        semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)

        async def get_msg_embedding(item):
            async with semaphore:
                msg = item["msg"]
                msg_content = msg.get("content", "")
                msg_text = self.extract_text_from_content(msg_content)

                content_key = msg.get("_content_key")
                if content_key and self.embedding_cache:
                    cached_embedding = self.embedding_cache.get(content_key)
                    if cached_embedding:
                        self.stats.cache_hits += 1
                        return item["idx"], cached_embedding

                self.stats.cache_misses += 1

                if self.has_images_in_content(msg_content):
                    msg_vector = await self.get_multimodal_embedding(
                        msg_content, progress.event_emitter
                    )
                    if not msg_vector:
                        msg_vector = await self.get_text_embedding(
                            msg_text, progress.event_emitter
                        )
                else:
                    msg_vector = await self.get_text_embedding(
                        msg_text, progress.event_emitter
                    )

                if content_key and msg_vector and self.embedding_cache:
                    self.embedding_cache.set(content_key, msg_vector)

                return item["idx"], msg_vector

        self.stats.concurrent_tasks = len(msg_items)
        embedding_tasks = [get_msg_embedding(item) for item in msg_items]
        embedding_results = await asyncio.gather(
            *embedding_tasks, return_exceptions=True
        )

        scored = []
        for item in msg_items:
            msg_vector = None
            for result in embedding_results:
                if isinstance(result, Exception):
                    continue
                result_idx, vector = result
                if result_idx == item["idx"]:
                    msg_vector = vector
                    break

            msg = item["msg"]
            msg_text = self.extract_text_from_content(msg.get("content", ""))

            sim = (
                self.cosine_similarity(query_vector, msg_vector)
                if (query_vector and msg_vector)
                else 0.0
            )

            recency = item["idx"] / max(1, len(msg_items) - 1)
            role = msg.get("role", "")
            role_weight = (
                1.0 if role == "user" else (0.8 if role == "assistant" else 0.6)
            )
            kw_bonus = 1.0 if self.is_high_priority_content(msg_text) else 0.0

            score = 0.6 * sim + 0.2 * recency + 0.1 * role_weight + 0.1 * kw_bonus

            scored.append(
                {
                    "msg": msg,
                    "score": score,
                    "tokens": item["tokens"],
                    "idx": item["idx"],
                    "sim": sim,
                    "recency": recency,
                    "role_weight": role_weight,
                    "kw_bonus": kw_bonus,
                }
            )

        return scored

    # ========== 升级策略 ==========

    def select_preserve_upgrades_with_protection(
        self, scored_msgs: List[dict], coverage_entries: List[dict], total_budget: int
    ) -> Tuple[set, int]:
        """选择升级的消息"""
        upgrade_pool = int(total_budget * self.valves.upgrade_min_pct)
        if upgrade_pool <= 0 or not scored_msgs:
            return set(), 0

        self.debug_log(
            1,
            f"升级池保护: 预留{upgrade_pool:,}tokens({self.valves.upgrade_min_pct:.1%})给升级",
            "⬆️",
        )

        summary_cost_map = defaultdict(int)
        for entry in coverage_entries:
            if entry["type"] == "micro":
                summary_cost_map[entry["msg_id"]] = entry.get(
                    "budget", entry.get("ideal_budget", 0)
                )

        candidates = []
        for item in scored_msgs:
            msg = item["msg"]
            msg_id = msg.get("_order_id", f"msg_{item['idx']}")
            original_tokens = item["tokens"]
            summary_cost = summary_cost_map.get(msg_id, 0)

            if summary_cost > 0:
                upgrade_cost = max(0, original_tokens - summary_cost)
            else:
                upgrade_cost = original_tokens

            if upgrade_cost <= 0:
                continue

            score = item["score"]
            if item["recency"] > 0.8:
                recency_boost = min(1.2, 1.0 + 0.2 * (2000 / max(upgrade_cost, 1)))
                score *= recency_boost

            density = score / upgrade_cost
            candidates.append(
                {
                    "density": density,
                    "score": score,
                    "upgrade_cost": upgrade_cost,
                    "item": item,
                    "msg_id": msg_id,
                }
            )

        candidates.sort(key=lambda x: (-x["density"], -x["score"]))

        preserve_set = set()
        consumed = 0
        self.debug_log(
            2, f"升级候选: {len(candidates)}个，升级池预算{upgrade_pool:,}tokens", "⬆️"
        )

        for cand in candidates:
            if consumed + cand["upgrade_cost"] > upgrade_pool:
                continue
            preserve_set.add(cand["msg_id"])
            consumed += cand["upgrade_cost"]
            self.debug_log(
                3,
                f"升级选中: ID={cand['msg_id'][:8]}, 密度={cand['density']:.4f}, 成本={cand['upgrade_cost']}tokens",
                "⬆️",
            )

        self.debug_log(
            1,
            f"升级选择完成: {len(preserve_set)}条消息升级, 消耗{consumed:,}/{upgrade_pool:,}tokens",
            "⬆️",
        )
        return preserve_set, consumed

    # ========== 摘要生成 ==========

    async def generate_micro_summary_with_budget_impl(
        self, msg: dict, budget: int, event_emitter
    ):
        """生成单条消息的微摘要"""
        client = self.get_api_client()
        if not client:
            return None

        content = self.extract_text_from_content(msg.get("content", ""))
        role = msg.get("role", "")
        cleaned_content = self.input_cleaner.clean_text_for_regex(content)

        prompt = f"""请为以下消息生成简洁摘要，保留关键信息。要求：
1. 严格在{budget}个tokens以内
2. 保留时间、主体、动作、数据/代码关键行等核心要素
3. 如果是技术内容，保留技术术语和关键参数
4. 保持客观简洁
消息角色: {role}
消息内容: {cleaned_content[:2000]}
摘要："""

        has_multimodal = self.has_images_in_content(msg.get("content"))
        model_to_use = (
            self.valves.multimodal_model if has_multimodal else self.valves.text_model
        )
        self.stats.summary_requests += 1

        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.2,
            timeout=self.valves.request_timeout,
        )

        if response.choices and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            summary = self.input_cleaner.clean_text_for_regex(summary)
            return summary
        return None

    async def generate_adaptive_block_summary_impl(
        self, msgs: List[dict], idx_range: Tuple[int, int], budget: int, event_emitter
    ):
        """生成自适应块摘要"""
        client = self.get_api_client()
        if not client:
            return None

        combined_content = ""
        has_multimodal = False
        for i, msg in enumerate(msgs):
            role = msg.get("role", "")
            content = self.extract_text_from_content(msg.get("content", ""))
            combined_content += f"[消息{idx_range[0] + i}:{role}] {content}\n\n"
            if self.has_images_in_content(msg.get("content")):
                has_multimodal = True

        cleaned_content = self.input_cleaner.clean_text_for_regex(combined_content)

        prompt = f"""请为以下连续消息块(第{idx_range[0]}到{idx_range[1]}条)生成综合摘要。要求：
1. 严格在{budget}个tokens以内
2. 覆盖所有要点，保持逻辑顺序
3. 指明消息编号范围和主要角色
4. 保留关键技术细节、数据、参数等
消息块内容：
{cleaned_content[:4000]}
块摘要："""

        model_to_use = (
            self.valves.multimodal_model if has_multimodal else self.valves.text_model
        )
        self.stats.summary_requests += 1

        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.2,
            timeout=self.valves.request_timeout,
        )

        if response.choices and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            summary = self.input_cleaner.clean_text_for_regex(summary)
            return summary
        return None

    async def generate_global_block_summary_impl(
        self, msgs: List[dict], idx_range: Tuple[int, int], budget: int, event_emitter
    ):
        """生成全局块摘要"""
        client = self.get_api_client()
        if not client:
            return None

        sampled_msgs = msgs[:: max(1, len(msgs) // 10)]
        combined_content = ""
        has_multimodal = False
        for i, msg in enumerate(sampled_msgs):
            role = msg.get("role", "")
            content = self.extract_text_from_content(msg.get("content", ""))
            combined_content += f"[消息样本{i}:{role}] {content[:200]}...\n\n"
            if self.has_images_in_content(msg.get("content")):
                has_multimodal = True

        cleaned_content = self.input_cleaner.clean_text_for_regex(combined_content)

        prompt = f"""请为以下对话历史生成全局摘要。要求：
1. 严格在{budget}个tokens以内
2. 概括主要话题和讨论要点
3. 保留重要的技术细节和结论
4. 总共涵盖{len(msgs)}条历史消息
对话历史样本：
{cleaned_content[:5000]}
全局摘要："""

        model_to_use = (
            self.valves.multimodal_model if has_multimodal else self.valves.text_model
        )
        self.stats.summary_requests += 1

        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.3,
            timeout=self.valves.request_timeout,
        )

        if response.choices and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            summary = self.input_cleaner.clean_text_for_regex(summary)
            return summary
        return None

    async def generate_coverage_summaries_with_budgets(
        self, coverage_entries: List[dict], progress: ProgressTracker
    ) -> Dict[str, str]:
        """并发生成覆盖摘要"""
        if not coverage_entries:
            return {}

        self.debug_log(1, f"开始并发生成覆盖摘要: {len(coverage_entries)}个条目", "📝")

        summaries = {}
        semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)

        async def generate_single_summary(entry):
            async with semaphore:
                if entry["type"] == "micro":
                    msg = entry["msg"]
                    budget = entry.get(
                        "budget",
                        entry.get(
                            "ideal_budget", self.valves.coverage_high_summary_tokens
                        ),
                    )
                    msg_id = entry["msg_id"]
                    summary = await self.safe_api_call(
                        self.generate_micro_summary_with_budget_impl,
                        "微摘要生成",
                        msg,
                        budget,
                        progress.event_emitter,
                    )
                    if summary:
                        self.stats.coverage_micro_summaries += 1
                        return msg_id, summary
                    else:
                        content = self.extract_text_from_content(msg.get("content", ""))
                        fallback_summary = (
                            content[: budget * 3] + "..."
                            if len(content) > budget * 3
                            else content
                        )
                        self.stats.guard_b_fallbacks += 1
                        return msg_id, f"[简化摘要] {fallback_summary}"

                elif entry["type"] == "adaptive_block":
                    msgs = entry["msgs"]
                    idx_range = entry["idx_range"]
                    budget = entry.get(
                        "budget",
                        entry.get(
                            "ideal_budget", self.valves.coverage_block_summary_tokens
                        ),
                    )
                    block_key = entry["block_key"]
                    summary = await self.safe_api_call(
                        self.generate_adaptive_block_summary_impl,
                        "自适应块摘要生成",
                        msgs,
                        idx_range,
                        budget,
                        progress.event_emitter,
                    )
                    if summary:
                        self.stats.coverage_block_summaries += 1
                        self.stats.adaptive_blocks_created += 1
                        return block_key, summary
                    else:
                        combined = " ".join(
                            [
                                f"[{msg.get('role','')}]{self.extract_text_from_content(msg.get('content',''))[:100]}..."
                                for msg in msgs
                            ]
                        )
                        self.stats.guard_b_fallbacks += 1
                        return (
                            block_key,
                            f"[简化块摘要] 第{idx_range[0]}-{idx_range[1]}条: {combined}",
                        )

                elif entry["type"] == "global_block":
                    msgs = entry["msgs"]
                    idx_range = entry["idx_range"]
                    budget = entry.get("budget", self.valves.min_block_summary_tokens)
                    block_key = entry["block_key"]
                    summary = await self.safe_api_call(
                        self.generate_global_block_summary_impl,
                        "全局块摘要生成",
                        msgs,
                        idx_range,
                        budget,
                        progress.event_emitter,
                    )
                    if summary:
                        self.stats.coverage_block_summaries += 1
                        return block_key, summary
                    else:
                        self.stats.guard_b_fallbacks += 1
                        return (
                            block_key,
                            f"[全局简化摘要] 包含{len(msgs)}条历史消息的对话内容",
                        )

                return None, None

        tasks = [generate_single_summary(entry) for entry in coverage_entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.stats.api_failures += 1
                continue
            key, summary = result
            if key and summary:
                summaries[key] = summary

        self.debug_log(1, f"并发摘要生成完成: {len(summaries)}个摘要", "📝")
        return summaries

    # ========== 组装阶段双重护栏 ==========

    async def assemble_coverage_output_with_guards(
        self,
        history_messages: List[dict],
        preserve_set: set,
        coverage_entries: List[dict],
        summaries: Dict[str, str],
        progress: ProgressTracker,
    ) -> List[dict]:
        """组装最终输出（双重护栏版本）"""
        if not history_messages:
            return []

        self.debug_log(1, f"开始组装最终输出: {len(history_messages)}条历史消息", "🔧")

        micro_entries = [e for e in coverage_entries if e["type"] == "micro"]
        adaptive_block_entries = [
            e for e in coverage_entries if e["type"] == "adaptive_block"
        ]
        global_block_entries = [
            e for e in coverage_entries if e["type"] == "global_block"
        ]

        if self.valves.debug_level >= 2:
            print(f"🛡️ 护栏A统计:")
            print(f"    ├─ 原文保留集合: {len(preserve_set)}条")
            print(f"    ├─ 微摘要条目: {len(micro_entries)}条")
            print(f"    ├─ 自适应块条目: {len(adaptive_block_entries)}条")
            print(f"    ├─ 全局块条目: {len(global_block_entries)}条")
            print(f"    ├─ 生成摘要总数: {len(summaries)}个")
            print(f"    └─ 历史消息总数: {len(history_messages)}条")

        all_micro_msg_ids = {e["msg_id"] for e in micro_entries}
        all_msg_ids = {
            msg.get("_order_id", f"msg_{i}") for i, msg in enumerate(history_messages)
        }
        unmapped_msg_ids = all_msg_ids - all_micro_msg_ids

        if unmapped_msg_ids and self.valves.debug_level >= 2:
            unmapped_sample = list(unmapped_msg_ids)[:3]
            print(
                f"🛡️ 护栏A警告: {len(unmapped_msg_ids)}条消息未映射到微摘要: {unmapped_sample}..."
            )
            self.stats.guard_a_warnings += 1

        msg_id_to_msg = {
            msg.get("_order_id", f"msg_{i}"): msg
            for i, msg in enumerate(history_messages)
        }

        block_summaries = {}
        block_ranges = {}
        entry_idx_ranges = {}

        for entry in adaptive_block_entries + global_block_entries:
            idx_range = entry["idx_range"]
            block_key = entry.get("block_key", f"block_{idx_range[0]}_{idx_range[1]}")
            entry_idx_ranges[block_key] = idx_range

            if block_key in summaries:
                block_summaries[block_key] = summaries[block_key]
                for idx in range(idx_range[0], idx_range[1] + 1):
                    if idx < len(history_messages):
                        block_ranges[idx] = block_key

        covered_by_micro_or_preserve = set()
        for i, msg in enumerate(history_messages):
            mid = msg.get("_order_id", f"msg_{i}")
            if mid in preserve_set or mid in summaries:
                covered_by_micro_or_preserve.add(i)

        final_messages = []
        processed_block_keys = set()
        covered_messages = 0

        for idx, msg in enumerate(history_messages):
            msg_id = msg.get("_order_id", f"msg_{idx}")
            message_covered = False

            if msg_id in preserve_set:
                final_messages.append(msg)
                self.stats.coverage_preserved_count += 1
                self.stats.coverage_preserved_tokens += self.count_message_tokens(msg)
                self.debug_log(3, f"使用原文: {msg_id[:8]}", "📄")
                message_covered = True

            elif msg_id in summaries:
                summary_msg = {
                    "role": "assistant",
                    "content": summaries[msg_id],
                    "_is_summary": True,
                    "_original_msg_id": msg_id,
                    "_summary_type": "micro",
                }
                final_messages.append(summary_msg)
                self.stats.coverage_summary_count += 1
                self.stats.coverage_summary_tokens += self.count_message_tokens(
                    summary_msg
                )
                self.debug_log(3, f"使用微摘要: {msg_id[:8]}", "📄")
                message_covered = True

            elif idx in block_ranges:
                block_key = block_ranges[idx]
                if (
                    block_key not in processed_block_keys
                    and block_key in block_summaries
                ):
                    idx0, idx1 = entry_idx_ranges[block_key]
                    has_uncovered = any(
                        j not in covered_by_micro_or_preserve
                        for j in range(idx0, idx1 + 1)
                        if j < len(history_messages)
                    )
                    if has_uncovered:
                        block_summary_msg = {
                            "role": "assistant",
                            "content": block_summaries[block_key],
                            "_is_summary": True,
                            "_block_key": block_key,
                            "_summary_type": (
                                "adaptive_block"
                                if "global" not in block_key
                                else "global_block"
                            ),
                        }
                        final_messages.append(block_summary_msg)
                        processed_block_keys.add(block_key)
                        self.stats.coverage_summary_count += 1
                        self.stats.coverage_summary_tokens += self.count_message_tokens(
                            block_summary_msg
                        )
                        self.debug_log(3, f"使用块摘要: {block_key}", "📄")
                        for j in range(idx0, idx1 + 1):
                            if j < len(history_messages):
                                message_covered = True
                                break

                if idx in block_ranges:
                    message_covered = True

            else:
                self.debug_log(
                    1, f"护栏B触发：消息{msg_id[:8]}既不在preserve也不在coverage中", "🛡️"
                )
                content = self.extract_text_from_content(msg.get("content", ""))
                fallback_msg = {
                    "role": "assistant",
                    "content": f"[护栏B简化摘要] {content[:200]}...",
                    "_is_summary": True,
                    "_original_msg_id": msg_id,
                    "_summary_type": "guard_b_fallback",
                }
                final_messages.append(fallback_msg)
                self.stats.guard_b_fallbacks += 1
                self.stats.coverage_summary_count += 1
                self.stats.coverage_summary_tokens += self.count_message_tokens(
                    fallback_msg
                )
                message_covered = True

            if message_covered:
                covered_messages += 1

        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        self.stats.coverage_total_messages = len(history_messages)
        self.stats.coverage_rate = covered_messages / max(1, len(history_messages))

        final_tokens = self.count_messages_tokens(final_messages)

        if self.valves.debug_level >= 2:
            print(f"🛡️ 护栏A最终验证:")
            print(
                f"    ├─ 最终消息数: 原文{self.stats.coverage_preserved_count}条 + 摘要{self.stats.coverage_summary_count}条 = {len(final_messages)}条"
            )
            print(
                f"    ├─ 覆盖率验证: {self.stats.coverage_rate:.1%} ({covered_messages}/{len(history_messages)})"
            )
            print(f"    └─ 最终token统计: {final_tokens:,}tokens")

        self.debug_log(
            1,
            f"双重护栏组装完成: {len(history_messages)} -> {len(final_messages)}条消息({final_tokens:,}tokens)",
            "✅",
        )
        return final_messages

    # ========== Top-up窗口填充器 ==========

    def topup_fill_window(
        self,
        final_messages: List[dict],
        scored_msgs: List[dict],
        available_tokens: int,
        summaries: Dict[str, str],
        preserve_set: set,
    ) -> List[dict]:
        """Top-up填充器"""
        initial_tokens = self.count_messages_tokens(final_messages)
        current_tokens = initial_tokens
        target_tokens = int(available_tokens * self.valves.target_window_usage)

        if current_tokens >= target_tokens:
            self.debug_log(
                1,
                f"窗口利用率已达标: {current_tokens:,}/{target_tokens:,} tokens ({self.valves.target_window_usage:.1%})",
                "🔥",
            )
            return final_messages

        self.debug_log(
            1,
            f"开始Top-up填充: {current_tokens:,} -> {target_tokens:,} tokens (目标{self.valves.target_window_usage:.1%})",
            "🔥",
        )
        self.stats.topup_applied += 1

        taken_micro = {
            m.get("_original_msg_id")
            for m in final_messages
            if m.get("_summary_type") == "micro"
        }
        id2msg = {
            item["msg"].get("_order_id", f"msg_{item['idx']}"): item
            for item in scored_msgs
        }

        micro_ids_sorted = sorted(
            [mid for mid in taken_micro if mid in id2msg],
            key=lambda mid: id2msg[mid]["score"] / max(1, id2msg[mid]["tokens"]),
            reverse=True,
        )

        upgraded_count = 0
        for mid in micro_ids_sorted:
            item = id2msg[mid]
            raw_msg = item["msg"]
            raw_tokens = self.count_message_tokens(raw_msg)

            micro_msg = None
            for i, msg in enumerate(final_messages):
                if msg.get("_original_msg_id") == mid:
                    micro_msg = msg
                    break

            if not micro_msg:
                continue

            micro_tokens = self.count_message_tokens(micro_msg)
            token_diff = raw_tokens - micro_tokens

            if current_tokens + token_diff > available_tokens:
                continue

            final_messages = [
                m for m in final_messages if m.get("_original_msg_id") != mid
            ]
            final_messages.append(raw_msg)
            current_tokens += token_diff
            upgraded_count += 1
            self.stats.topup_micro_upgraded += 1
            self.debug_log(
                3, f"微摘要升级为原文: {mid[:8]}, 增加{token_diff}tokens", "⬆️"
            )

            if current_tokens >= target_tokens:
                break

        if upgraded_count > 0:
            self.debug_log(1, f"微摘要升级完成: {upgraded_count}条升级", "⬆️")

        landed_ids = {
            m.get("_order_id") or m.get("_original_msg_id") for m in final_messages
        }
        candidates = [
            it for it in scored_msgs if it["msg"].get("_order_id") not in landed_ids
        ]
        candidates.sort(key=lambda it: it["score"] / max(1, it["tokens"]), reverse=True)

        added_count = 0
        for item in candidates:
            tokens = item["tokens"]
            if current_tokens + tokens > available_tokens:
                continue
            final_messages.append(item["msg"])
            current_tokens += tokens
            added_count += 1
            self.stats.topup_raw_added += 1
            self.debug_log(
                3,
                f"添加未落地原文: {item['msg'].get('_order_id', 'unknown')[:8]}, 增加{tokens}tokens",
                "📝",
            )
            if current_tokens >= target_tokens:
                break

        if added_count > 0:
            self.debug_log(1, f"未落地原文添加完成: {added_count}条添加", "📝")

        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        final_tokens = self.count_messages_tokens(final_messages)
        tokens_added = max(0, final_tokens - initial_tokens)
        self.stats.topup_tokens_added += tokens_added
        utilization = final_tokens / available_tokens if available_tokens > 0 else 0

        self.debug_log(
            1,
            f"Top-up填充完成: {final_tokens:,}tokens, 利用率{utilization:.1%}, 新增{tokens_added:,}tokens",
            "✅",
        )
        return final_messages

    # ========== 上下文最大化主流程 ==========

    async def process_coverage_first_context_maximization_v2(
        self,
        history_messages: List[dict],
        available_tokens: int,
        progress: ProgressTracker,
        query_message: dict,
        allow_topup: bool = False,
    ) -> List[dict]:
        """上下文最大化上下文最大化处理主流程"""
        if not history_messages or not self.valves.enable_coverage_first:
            return history_messages

        await progress.start_phase("上下文最大化处理", len(history_messages))
        self.debug_log(
            1,
            f"上下文最大化开始: {len(history_messages)}条消息, 可用预算: {available_tokens:,}tokens",
            "🎯",
        )

        if self.valves.enable_smart_chunking:
            await progress.update_progress(0, 8, "消息分片预处理")
            processed_history = self.message_chunker.preprocess_messages_with_chunking(
                history_messages, self.message_order
            )
            self.stats.chunked_messages_count = len(
                [msg for msg in processed_history if msg.get("_is_chunk")]
            )
            self.stats.total_chunks_created = sum(
                1 for m in processed_history if m.get("_is_chunk")
            )
            self.debug_log(
                1,
                f"消息分片预处理: {len(history_messages)} -> {len(processed_history)}条 ({self.stats.chunked_messages_count}条被分片)",
                "🧩",
            )
        else:
            processed_history = history_messages

        await progress.update_progress(1, 8, "计算相关度分数")
        scored_msgs = await self.compute_relevance_scores(
            query_message, processed_history, progress
        )

        if not scored_msgs:
            self.debug_log(1, "相关度计算失败，使用原始消息", "⚠️")
            return processed_history

        await progress.update_progress(2, 8, "自适应Coverage规划")
        upgrade_pool = int(available_tokens * self.valves.upgrade_min_pct)
        coverage_budget = available_tokens - upgrade_pool
        coverage_entries, coverage_cost = (
            self.coverage_planner.plan_adaptive_coverage_summaries(
                scored_msgs, coverage_budget
            )
        )

        if coverage_cost < coverage_budget:
            actual_upgrade_pool = upgrade_pool + (coverage_budget - coverage_cost)
        else:
            actual_upgrade_pool = upgrade_pool

        if coverage_cost != coverage_budget:
            self.stats.budget_scaling_applied += 1
            self.stats.scaling_factor = (
                coverage_cost / coverage_budget if coverage_budget > 0 else 1.0
            )

        self.debug_log(
            1,
            f"自适应Coverage规划: {len(coverage_entries)}个条目, 成本{coverage_cost:,}tokens (升级池{actual_upgrade_pool:,}tokens)",
            "📄",
        )

        await progress.update_progress(3, 8, "升级策略选择")
        preserve_set, upgrade_consumed = self.select_preserve_upgrades_with_protection(
            scored_msgs, coverage_entries, actual_upgrade_pool
        )

        self.stats.coverage_upgrade_count = len(preserve_set)
        self.stats.coverage_upgrade_tokens_saved = upgrade_consumed

        await progress.update_progress(4, 8, "并发生成摘要内容")
        summaries = await self.generate_coverage_summaries_with_budgets(
            coverage_entries, progress
        )

        await progress.update_progress(5, 8, "双重护栏组装")
        final_messages = await self.assemble_coverage_output_with_guards(
            processed_history, preserve_set, coverage_entries, summaries, progress
        )

        if allow_topup and self.valves.enable_window_topup:
            await progress.update_progress(6, 8, "Top-up窗口填充")
            final_messages = self.topup_fill_window(
                final_messages, scored_msgs, available_tokens, summaries, preserve_set
            )

        await progress.update_progress(7, 8, "最终统计计算")
        final_tokens = self.count_messages_tokens(final_messages)
        self.stats.coverage_budget_usage = (
            final_tokens / available_tokens if available_tokens > 0 else 0
        )

        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        await progress.update_progress(8, 8, "处理完成")
        self.debug_log(
            1,
            f"上下文最大化完成: {len(processed_history)} -> {len(final_messages)}条消息",
            "✅",
        )
        self.debug_log(
            1,
            f"统计: 覆盖率{self.stats.coverage_rate:.1%}, 预算使用{self.stats.coverage_budget_usage:.1%}",
            "✅",
        )

        await progress.complete_phase(
            f"覆盖率{self.stats.coverage_rate:.1%} 预算使用{self.stats.coverage_budget_usage:.1%}"
        )
        return final_messages

    # ========== 视觉处理 ==========

    def validate_base64_image_data(self, image_data: str) -> bool:
        """验证base64图片数据的有效性"""
        return self.input_cleaner.validate_and_clean_image_url(image_data)[0]

    async def describe_image_impl(self, image_data: str, event_emitter):
        """实际的图片描述实现"""
        client = self.get_api_client()
        if not client:
            return None

        is_valid, cleaned_data = self.input_cleaner.validate_and_clean_image_url(
            image_data
        )
        if not is_valid:
            self.debug_log(1, "图片数据验证失败", "⚠️")
            self.stats.image_processing_errors += 1
            return "图片格式错误：不是有效的URL或data URI"

        try:
            response = await client.chat.completions.create(
                model=self.valves.multimodal_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.valves.vision_prompt_template,
                            },
                            {"type": "image_url", "image_url": {"url": cleaned_data}},
                        ],
                    }
                ],
                max_tokens=self.valves.vision_max_tokens,
                temperature=0.2,
                timeout=self.valves.request_timeout,
            )

            if response.choices and response.choices[0].message.content:
                description = response.choices[0].message.content.strip()
                description = self.input_cleaner.clean_text_for_regex(description)
                return description
            else:
                self.stats.image_processing_errors += 1
                return "图片识别失败：API返回空响应"
        except Exception as e:
            await self.learn_model_capability_from_errors(
                self.valves.multimodal_model, error_text=str(e)
            )
            self.debug_log(1, f"图片识别异常: {str(e)[:100]}", "❌")
            self.stats.image_processing_errors += 1
            return f"图片识别失败：{str(e)[:100]}"

    async def describe_image(self, image_data: str, event_emitter) -> str:
        """描述单张图片"""
        if not image_data:
            return "图片数据为空"

        description = await self.safe_api_call(
            self.describe_image_impl,
            "图片识别",
            image_data,
            event_emitter,
        )

        if description:
            if len(description) > 3000:
                description = description[:3000] + "..."
            return description
        else:
            self.stats.image_processing_errors += 1
            return "图片处理失败：无法获取描述"

    async def process_message_images(
        self, message: dict, progress: "ProgressTracker"
    ) -> dict:
        """处理单条消息中的图片"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message

        images = [item for item in content if item.get("type") == "image_url"]
        if not images:
            return message

        self.debug_log(2, f"处理消息中的图片: {len(images)}张", "🖼️")

        processed_content = []
        image_count = 0
        image_meta = []

        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text.strip():
                    processed_content.append(text)
            elif item.get("type") == "image_url":
                image_count += 1
                image_data = item.get("image_url", {}).get("url", "")

                is_valid, cleaned = self.input_cleaner.validate_and_clean_image_url(
                    image_data
                )
                if not is_valid:
                    self.stats.image_processing_errors += 1
                    processed_content.append(f"[图片{image_count}无法识别]")
                    continue

                if progress:
                    await progress.update_progress(
                        image_count,
                        len(images),
                        f"处理图片 {image_count}/{len(images)}",
                    )

                image_hash = hashlib.md5(cleaned.encode()).hexdigest()
                if image_hash in self.image_description_cache:
                    description = self.image_description_cache[image_hash]
                    self.debug_log(2, f"复用图片描述缓存: {image_hash[:8]}", "♻️")
                else:
                    description = await self.describe_image(
                        cleaned, progress.event_emitter if progress else None
                    )
                    self.image_description_cache[image_hash] = description

                image_name = f"img_{image_hash[:8]}"
                image_line = f"[图片{image_count} {image_name}] {description}"
                processed_content.append(image_line)

                image_meta.append(
                    {
                        "index": image_count,
                        "name": image_name,
                        "source": "user",
                        "url": cleaned,
                    }
                )

        if image_count == 0:
            return message

        processed_message = copy.deepcopy(message)
        processed_message["content"] = (
            "\n".join(processed_content) if processed_content else ""
        )
        processed_message["_images_processed"] = image_count
        if image_meta:
            existing_meta = processed_message.get("_image_meta") or []
            processed_message["_image_meta"] = existing_meta + image_meta

        self.stats.multimodal_processed += image_count
        return processed_message

    def strip_images_from_message(self, message: dict) -> dict:
        """将历史消息中的图片替换为占位符"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message

        processed_parts = []
        image_count = 0
        meta_list = message.get("_image_meta") or []
        name_by_index = {
            m.get("index"): m.get("name")
            for m in meta_list
            if m.get("index") is not None
        }

        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text.strip():
                    processed_parts.append(text)
            elif item.get("type") == "image_url":
                image_count += 1
                tag_name = name_by_index.get(image_count)
                if tag_name:
                    processed_parts.append(f"[历史图片{image_count} {tag_name}]")
                else:
                    processed_parts.append(f"[历史图片{image_count}]")

        processed_message = copy.deepcopy(message)
        if processed_parts:
            processed_message["content"] = "\n".join(processed_parts)
        else:
            processed_message["content"] = "[历史图片已省略，请参考对话中的文字说明]"
        processed_message["_images_processed"] = image_count
        return processed_message

    # ========== 多模态处理策略 ==========

    def calculate_multimodal_budget_sufficient(
        self, messages: List[dict], target_tokens: int
    ) -> bool:
        """计算多模态模型的Token预算是否充足"""
        current_tokens = self.count_messages_tokens(messages)
        usage_ratio = current_tokens / target_tokens if target_tokens > 0 else 1.0
        threshold = self.valves.multimodal_direct_threshold
        is_sufficient = usage_ratio <= threshold
        self.debug_log(
            1,
            f"多模态预算检查: {current_tokens:,}/{target_tokens:,} = {usage_ratio:.2%} {'≤' if is_sufficient else '>'} {threshold:.1%}",
            "💰",
        )
        return is_sufficient

    async def determine_multimodal_processing_strategy(
        self, messages: List[dict], model_name: str, target_tokens: int
    ) -> Tuple[str, str]:
        """确定多模态处理策略"""
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return "text_only", "无图片内容，按文本处理"

        known_non_multimodal = self._is_model_known_non_multimodal(model_name)
        is_multimodal = self.is_multimodal_model(model_name)

        if not known_non_multimodal:
            probe_result = await self.probe_model_multimodal_support(model_name)
            if probe_result is False:
                known_non_multimodal = True
            elif probe_result is True:
                is_multimodal = True
            elif not is_multimodal:
                # 探测未得出结论且当前默认判定为文本模型时，不再盲目直通多模态
                known_non_multimodal = True

        self.debug_log(
            1,
            f"模型分析: {model_name} | 多模态支持: {is_multimodal} | 已知不支持: {known_non_multimodal}",
            "🤖",
        )

        if known_non_multimodal:
            return "vision_to_text", "模型探测/学习后确认不支持多模态，使用图片描述文本"

        if is_multimodal:
            budget_sufficient = self.calculate_multimodal_budget_sufficient(
                messages, target_tokens
            )
            if budget_sufficient:
                return "direct_multimodal", "多模态模型，Token预算充足，直接输入"
            else:
                return "multimodal_rag", "多模态模型，Token预算不足，使用多模态向量RAG"

        return "direct_multimodal", "模型能力未知但探测未判定不支持，先直通"

    async def process_multimodal_content(
        self,
        messages: List[dict],
        model_name: str,
        target_tokens: int,
        progress: "ProgressTracker",
    ) -> List[dict]:
        """多模态内容处理"""
        if not self.valves.enable_multimodal:
            return messages

        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages

        strategy, strategy_desc = await self.determine_multimodal_processing_strategy(
            messages, model_name, target_tokens
        )
        self.debug_log(1, f"多模态策略: {strategy} - {strategy_desc}", "🎯")

        if strategy == "text_only":
            return messages
        elif strategy == "direct_multimodal":
            return messages
        elif strategy == "vision_to_text":
            await progress.start_phase("视觉识别转文本", 1)
            current_index = (
                self.message_order.find_current_user_message_index(messages)
                if self.message_order
                else -1
            )

            processed_messages: List[dict] = []
            for i, message in enumerate(messages):
                content = message.get("content")
                if not self.has_images_in_content(content):
                    processed_messages.append(message)
                    continue

                if i == current_index:
                    processed = await self.process_message_images(message, progress)
                    processed_messages.append(processed)
                else:
                    processed = self.strip_images_from_message(message)
                    processed_messages.append(processed)

            if self.message_order:
                processed_messages = self.message_order.sort_messages_preserve_user(
                    processed_messages, self.current_user_message
                )

            self._mark_model_as_text_only(
                model_name,
                "检测到该模型在当前会话已走视觉转文本流程，后续按不支持多模态处理。",
            )
            await progress.complete_phase("视觉识别完成")
            return processed_messages
        else:
            return messages

    # ========== 智能截断 ==========

    def smart_truncate_messages(
        self, messages: List[dict], target_tokens: int, preserve_priority: bool = True
    ) -> List[dict]:
        """智能截断算法"""
        if not messages:
            return messages

        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= target_tokens:
            return messages

        self.debug_log(
            1, f"开始智能截断: {current_tokens:,} -> {target_tokens:,}tokens", "✂️"
        )
        self.stats.smart_truncation_applied += 1

        if preserve_priority:
            message_priorities = []
            for i, msg in enumerate(messages):
                priority_score = self._calculate_message_priority(msg, i, len(messages))
                message_priorities.append((i, msg, priority_score))
            message_priorities.sort(key=lambda x: x[2], reverse=True)
        else:
            message_priorities = [(i, msg, 1.0) for i, msg in enumerate(messages)]

        selected_messages = []
        used_tokens = 0
        skipped_messages = []

        for original_idx, msg, priority in message_priorities:
            msg_tokens = self.count_message_tokens(msg)
            if used_tokens + msg_tokens <= target_tokens:
                selected_messages.append((original_idx, msg, priority))
                used_tokens += msg_tokens
            else:
                skipped_messages.append((original_idx, msg, priority, msg_tokens))
                self.stats.truncation_skip_count += 1

        remaining_budget = target_tokens - used_tokens
        if remaining_budget > 100 and skipped_messages:
            skipped_messages.sort(key=lambda x: x[3])
            recovered_count = 0
            for original_idx, msg, priority, msg_tokens in skipped_messages:
                if msg_tokens <= remaining_budget:
                    selected_messages.append((original_idx, msg, priority))
                    used_tokens += msg_tokens
                    remaining_budget -= msg_tokens
                    recovered_count += 1
                    if remaining_budget < 100:
                        break
            self.stats.truncation_recovered_messages += recovered_count

        selected_messages.sort(key=lambda x: x[0])
        final_messages = [msg for _, msg, _ in selected_messages]

        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        final_tokens = self.count_messages_tokens(final_messages)
        retention_ratio = len(final_messages) / len(messages) if messages else 0
        self.debug_log(
            1,
            f"智能截断完成: {len(messages)} -> {len(final_messages)}条消息 保留率{retention_ratio:.1%}",
            "✅",
        )
        return final_messages

    def _calculate_message_priority(
        self, msg: dict, index: int, total_count: int
    ) -> float:
        """计算消息优先级分数"""
        priority = 1.0

        role = msg.get("role", "")
        if role == "user":
            priority += 2.0
        elif role == "assistant":
            priority += 1.5
        elif role == "system":
            priority += 3.0

        position_score = index / total_count if total_count > 0 else 0
        priority += position_score * 2.0

        content_text = self.extract_text_from_content(msg.get("content", ""))
        if self.is_high_priority_content(content_text):
            priority += 1.5

        content_length = len(content_text)
        if 100 < content_length < 2000:
            priority += 0.5
        elif content_length > 5000:
            priority -= 1.0
        elif content_length > 10000:
            priority -= 2.0

        if self.has_images_in_content(msg.get("content")):
            priority += 1.0

        if msg.get("_is_summary"):
            priority += 0.8

        if msg.get("_is_chunk"):
            priority += 0.3

        return priority

    # ========== 用户消息保护 ==========

    def ensure_current_user_message_preserved(
        self, final_messages: List[dict]
    ) -> List[dict]:
        """确保当前用户消息被正确保留在最后位置"""
        if not self.current_user_message:
            return final_messages

        if final_messages and final_messages[-1].get("role") == "user":
            current_id = self.current_user_message.get("_order_id")
            last_id = final_messages[-1].get("_order_id")
            if current_id == last_id:
                return final_messages

        self.debug_log(1, "检测到当前用户消息位置错误，开始修复", "🛡️")
        current_id = self.current_user_message.get("_order_id")
        filtered_messages = []
        for msg in final_messages:
            if msg.get("_order_id") != current_id:
                filtered_messages.append(msg)
        filtered_messages.append(self.current_user_message)

        self.stats.user_message_recovery_count += 1
        self.debug_log(1, "当前用户消息位置修复完成", "🛡️")
        return filtered_messages

    # ========== 主要处理逻辑 ==========

    async def maximize_content_comprehensive_processing_v2(
        self, messages: List[dict], target_tokens: int, progress: ProgressTracker
    ) -> List[dict]:
        """内容最大化综合处理"""
        start_time = time.time()

        current_model_name = getattr(self, "_current_model_name", "unknown")
        if hasattr(self, "current_model_info") and self.current_model_info:
            model_limit = self.current_model_info.get(
                "limit", self.valves.default_token_limit
            )
            safe_limit = int(model_limit * self.valves.token_safety_ratio)
        else:
            safe_limit = self.get_model_token_limit(current_model_name)

        self.stats.original_tokens = self.count_messages_tokens(messages)
        self.stats.original_messages = len(messages)
        self.stats.token_limit = safe_limit
        self.stats.target_tokens = target_tokens
        current_tokens = self.stats.original_tokens

        self.debug_log(
            1,
            f"上下文最大化处理开始: {current_tokens:,} tokens, {len(messages)} 条消息",
            "🎯",
        )

        await progress.start_phase("上下文最大化处理", 10)

        await progress.update_progress(1, 10, "分离当前用户消息和历史消息")
        current_user_message, history_messages = (
            self.separate_current_and_history_messages(messages)
        )
        self.current_user_message = current_user_message

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        if current_user_message:
            self.stats.current_user_tokens = self.count_message_tokens(
                current_user_message
            )

        need_context_max = False
        if current_user_message and self.valves.enable_context_maximization:
            query_text = self.extract_text_from_content(
                current_user_message.get("content", "")
            )
            need_context_max = await self.detect_context_max_need(
                query_text, progress.event_emitter
            )
            if need_context_max:
                self.debug_log(1, f"检测到需要上下文最大化，启用上下文最大化策略", "📚")

        protected_messages = system_messages[:]
        protected_tokens = self.count_messages_tokens(protected_messages)
        available_for_processing = (
            target_tokens - protected_tokens - self.stats.current_user_tokens
        )

        self.debug_log(
            1, f"历史消息可用处理空间: {available_for_processing:,}tokens", "💰"
        )

        if not history_messages:
            final_messages = system_messages[:]
            if current_user_message:
                final_messages.append(current_user_message)
            await progress.complete_phase("无历史消息需要处理")
            return final_messages

        if (
            need_context_max
            and self.valves.enable_context_maximization
            and self.valves.enable_coverage_first
        ):
            await progress.update_progress(2, 10, "上下文最大化专用处理")
            processed_history = (
                await self.process_coverage_first_context_maximization_v2(
                    history_messages,
                    available_for_processing,
                    progress,
                    current_user_message,
                    allow_topup=True and self.valves.enable_window_topup,
                )
            )
        else:
            await progress.update_progress(2, 10, "标准截断处理")
            if available_for_processing > 0:
                processed_history = self.smart_truncate_messages(
                    history_messages, available_for_processing, True
                )
            else:
                processed_history = []

        await progress.update_progress(6, 10, "不截断保障检查")
        final_history = processed_history
        final_tokens = self.count_messages_tokens(final_history)

        if (
            final_tokens > available_for_processing
            and self.valves.disable_insurance_truncation
        ):
            self.debug_log(1, f"预算超限但禁用截断，保证不截断", "🛡️")
            self.stats.insurance_truncation_avoided += 1
        elif final_tokens > available_for_processing:
            self.debug_log(1, f"超出预算，启用保险截断", "✂️")
            final_history = self.smart_truncate_messages(
                final_history, available_for_processing, True
            )
            final_tokens = self.count_messages_tokens(final_history)
            self.stats.zero_loss_guarantee = False

        await progress.update_progress(8, 10, "组合最终结果")
        current_result = system_messages + final_history

        if self.message_order:
            current_result = self.message_order.sort_messages_preserve_user(
                current_result, self.current_user_message
            )

        final_messages = []
        for msg in current_result:
            final_messages.append(msg)
        if current_user_message:
            final_messages.append(current_user_message)

        await progress.update_progress(9, 10, "用户消息保护验证")
        final_messages = self.ensure_current_user_message_preserved(final_messages)

        await progress.update_progress(10, 10, "更新统计")
        self.stats.final_tokens = self.count_messages_tokens(final_messages)
        self.stats.final_messages = len(final_messages)
        self.stats.processing_time = time.time() - start_time
        self.stats.iterations = 1

        if self.stats.original_tokens > 0:
            self.stats.content_loss_ratio = max(
                0,
                (self.stats.original_tokens - self.stats.final_tokens)
                / self.stats.original_tokens,
            )

        if target_tokens > 0:
            self.stats.window_utilization = self.stats.final_tokens / target_tokens

        if current_user_message:
            self.stats.current_user_preserved = any(
                msg.get("_order_id") == current_user_message.get("_order_id")
                for msg in final_messages
            )

        retention_ratio = self.stats.calculate_retention_ratio()
        window_usage = self.stats.calculate_window_usage_ratio()

        self.debug_log(
            1,
            f"上下文最大化处理完成: 保留{retention_ratio:.1%} 窗口使用{window_usage:.1%} 不截断{'保障成功' if self.stats.zero_loss_guarantee else '部分失效'}",
            "🎯",
        )

        await progress.complete_phase(
            f"覆盖率{self.stats.coverage_rate:.1%} 预算使用{window_usage:.1%} "
            f"不截断{'成功' if self.stats.zero_loss_guarantee else '失效'} "
            f"{'[上下文最大化]' if need_context_max else '[具体查询]'}"
        )
        return final_messages

    def print_detailed_stats(self):
        """打印详细统计信息"""
        if not self.valves.enable_detailed_stats:
            return
        print("\n" + "=" * 60)
        print(self.stats.get_summary())
        print("=" * 60)

    # ========== 入口和出口函数 ==========

    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """入口函数"""
        if self.valves.debug_level >= 1:
            print("🚀 高级上下文管理器启动（自动记忆后台运行）")

        # 兼容不同版本传参：user / __user__
        user = (
            user
            if isinstance(user, dict)
            else (__user__ if isinstance(__user__, dict) else None)
        )
        if user is None and isinstance(kwargs.get("user"), dict):
            user = kwargs.get("user")
        if user is None and isinstance(kwargs.get("__user__"), dict):
            user = kwargs.get("__user__")
        try:
            if isinstance(user, dict) and user.get("id") is not None:
                self.current_user_obj = Users.get_user_by_id(user["id"])
        except Exception:
            pass

        # 1. 记忆上下文处理（如果启用）
        if self.valves.enable_auto_memory and self.valves.override_memory_context:
            if "messages" in body:
                try:
                    body["messages"] = self.process_memory_context_in_messages(
                        body["messages"]
                    )
                except Exception as e:
                    self.memory_log(f"记忆上下文处理失败: {e}", "error")

        # 2. 原有的上下文处理逻辑
        if not self.valves.enable_processing:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        model_name = body.get("model", "未知")
        if self.is_model_excluded(model_name):
            return body

        self.reset_processing_state()
        self._current_model_name = model_name
        self.current_model_info = self.analyze_model(model_name)

        original_tokens = self.count_messages_tokens(messages)
        model_token_limit = self.get_model_token_limit(model_name)
        current_user_tokens = (
            self.count_message_tokens(self.find_current_user_message(messages))
            if self.find_current_user_message(messages)
            else 0
        )
        target_tokens = self.calculate_target_tokens(model_name, current_user_tokens)

        needs_proc, token_overflow, mm_incompat = self._needs_processing(
            messages, model_name, target_tokens
        )

        show_progress = needs_proc or not self.valves.suppress_frontend_when_idle
        progress = ProgressTracker(__event_emitter__ if show_progress else None)

        self.message_order = MessageOrder(messages)
        messages = self.message_order.original_messages

        current_user_message, history_messages = (
            self.separate_current_and_history_messages(messages)
        )
        self.current_user_message = current_user_message

        self.stats.token_limit = model_token_limit
        self.stats.target_tokens = target_tokens
        self.stats.current_user_tokens = current_user_tokens

        if self.valves.debug_level >= 1:
            print(
                f"模型: {self.current_model_info['family']} | tokens: {original_tokens:,}/{model_token_limit:,} | 历史: {len(history_messages)}条"
            )

        if current_user_message:
            content_preview = self.message_order.get_message_preview(
                current_user_message
            )
            processing_id = hashlib.md5(
                f"{current_user_message.get('_order_id', '')}{content_preview}{time.time()}".encode()
            ).hexdigest()[:8]
            self.current_processing_id = processing_id

            need_context_max = False
            if False and self.valves.enable_ai_context_max_detection and needs_proc:
                query_text = self.extract_text_from_content(
                    current_user_message.get("content", "")
                )
                try:
                    need_context_max = await self.detect_context_max_need(
                        query_text, __event_emitter__
                    )
                    if self.valves.debug_level >= 1:
                        print(
                            f"上下文最大化检测: {'需要' if need_context_max else '不需要'}"
                        )
                except Exception as e:
                    if self.valves.debug_level >= 1:
                        print(f"AI检测失败: {e}")
                    need_context_max = self.is_context_max_need_simple(query_text)

        should_maximize = needs_proc

        try:
            if self.valves.enable_detailed_progress:
                await progress.start_phase("多模态处理", 1)

            processed_messages = await self.process_multimodal_content(
                messages, model_name, target_tokens, progress
            )
            processed_tokens = self.count_messages_tokens(processed_messages)

            if not self.is_multimodal_model(model_name):
                _post_tmp_user = self.find_current_user_message(processed_messages)
                if _post_tmp_user is not None:
                    c = _post_tmp_user.get("content")
                    if not isinstance(c, str):
                        parts = []
                        if isinstance(c, list):
                            for it in c:
                                if isinstance(it, str):
                                    parts.append(it)
                                elif isinstance(it, dict):
                                    t = it.get("type")
                                    if t == "text" and isinstance(it.get("text"), str):
                                        parts.append(it["text"])
                                    elif t == "image_url":
                                        img = it.get("image_url")
                                        url = (
                                            img.get("url", "")
                                            if isinstance(img, dict)
                                            else (img if isinstance(img, str) else "")
                                        )
                                        parts.append(
                                            f"[图片] {url}" if url else "[图片]"
                                        )
                                    elif isinstance(it.get("content"), str):
                                        parts.append(it["content"])
                        elif isinstance(c, dict):
                            if c.get("type") == "text" and isinstance(
                                c.get("text"), str
                            ):
                                parts.append(c["text"])
                            elif c.get("type") == "image_url":
                                img = c.get("image_url")
                                url = (
                                    img.get("url", "")
                                    if isinstance(img, dict)
                                    else (img if isinstance(img, str) else "")
                                )
                                parts.append(f"[图片] {url}" if url else "[图片]")
                            elif isinstance(c.get("content"), str):
                                parts.append(c["content"])
                        _post_tmp_user["content"] = "\n".join(
                            p for p in parts if isinstance(p, str)
                        ).strip()

            processed_tokens = self.count_messages_tokens(processed_messages)

            _post_user = self.find_current_user_message(processed_messages)
            _post_user_tokens = (
                self.count_message_tokens(_post_user) if _post_user else 0
            )
            target_tokens = self.calculate_target_tokens(model_name, _post_user_tokens)
            post_needs_proc, post_token_overflow, post_mm_incompat = (
                self._needs_processing(processed_messages, model_name, target_tokens)
            )

            if not post_needs_proc:
                self.stats.original_tokens = self.count_messages_tokens(messages)
                self.stats.original_messages = len(messages)
                self.stats.final_tokens = processed_tokens
                self.stats.final_messages = len(processed_messages)
                body["messages"] = copy.deepcopy(processed_messages)
                body["messages"] = self.strip_internal_fields(body["messages"])
                if self.valves.debug_level >= 1:
                    print("无需处理：多模态转写后未超限，直接返回原文（或转写后）")
                return body

            should_maximize = post_needs_proc
            if (
                self.valves.enable_ai_context_max_detection
                and should_maximize
                and _post_user
            ):
                query_text = self.extract_text_from_content(
                    _post_user.get("content", "")
                )
                try:
                    need_context_max = await self.detect_context_max_need(
                        query_text, __event_emitter__
                    )
                    if self.valves.debug_level >= 1:
                        print(
                            f"上下文最大化检测(后移): {'需要' if need_context_max else '不需要'}"
                        )
                    if not need_context_max:
                        should_maximize = False
                except Exception as e:
                    if self.valves.debug_level >= 1:
                        print(f"AI检测(后移)失败: {e}")

            if should_maximize:
                final_messages = (
                    await self.maximize_content_comprehensive_processing_v2(
                        processed_messages, target_tokens, progress
                    )
                )
                self.print_detailed_stats()
                body["messages"] = copy.deepcopy(final_messages)

                final_tokens = self.count_messages_tokens(final_messages)
                window_utilization = (
                    final_tokens / target_tokens if target_tokens > 0 else 0
                )

                if self.valves.debug_level >= 1:
                    print(
                        f"处理完成: {len(final_messages)}条消息, {final_tokens:,}tokens, 利用率{window_utilization:.1%}, 不截断{'✅' if self.stats.zero_loss_guarantee else '⚠️'}"
                    )

                if current_user_message and final_messages:
                    last_msg = final_messages[-1]
                    if last_msg.get("role") == "user" and self.valves.debug_level >= 1:
                        print(f"当前用户消息保护: ✅")
            else:
                self.stats.original_tokens = self.count_messages_tokens(messages)
                self.stats.original_messages = len(messages)
                self.stats.final_tokens = processed_tokens
                self.stats.final_messages = len(processed_messages)
                if self.valves.enable_detailed_progress:
                    await progress.complete_phase("无需最大化处理")
                body["messages"] = copy.deepcopy(processed_messages)
                if self.valves.debug_level >= 1:
                    print(f"直接使用处理后的消息")

        except Exception as e:
            print(f"❌ 处理异常: {e}")
            self.stats.api_failures += 1
            import traceback

            if self.valves.debug_level >= 2:
                traceback.print_exc()
            if self.valves.enable_detailed_progress:
                await progress.update_status(f"处理失败: {str(e)[:50]}", True)

        if self.valves.debug_level >= 1:
            print("🏁 上下文最大化处理完成")

        if isinstance(body.get("messages"), list):
            body["messages"] = self.strip_internal_fields(body["messages"])

        # ========== Auto Memory 兜底触发（兼容某些版本 outlet hook 不触发） ==========
        try:
            if getattr(self.valves, "enable_auto_memory", False):
                msgs = body.get("messages") or []
                if isinstance(msgs, list) and len(msgs) >= 1:
                    # 尽量避免处理“当前未完成的一轮”（通常 inlet 最后一个是本轮 user 消息）
                    cand = msgs
                    try:
                        if (
                            isinstance(msgs[-1], dict)
                            and msgs[-1].get("role") == "user"
                            and len(msgs) >= 2
                        ):
                            cand = msgs[:-1]
                            if len(cand) == 0:
                                cand = msgs
                    except Exception:
                        cand = msgs

                    # 去重：避免 inlet/outlet 双触发重复写入
                    import json as _am_json
                    import hashlib as _am_hashlib
                    import time as _am_time

                    try:
                        sig_src = _am_json.dumps(
                            cand[-4:], ensure_ascii=False, sort_keys=True
                        )
                    except Exception:
                        sig_src = str(cand[-4:])
                    sig = _am_hashlib.sha1(
                        sig_src.encode("utf-8", errors="ignore")
                    ).hexdigest()
                    last_sig = getattr(self, "_am_last_sig", None)
                    last_ts = getattr(self, "_am_last_ts", 0.0)
                    now_ts = _am_time.time()
                    if sig != last_sig or (now_ts - float(last_ts)) > 30.0:
                        setattr(self, "_am_last_sig", sig)
                        setattr(self, "_am_last_ts", now_ts)

                        # 兼容不同版本传参：user / __user__ / body["user"]
                        user_dict = user if isinstance(user, dict) else None
                        if user_dict is None and isinstance(body.get("__user__"), dict):
                            user_dict = body.get("__user__")
                        if user_dict is None and isinstance(body.get("user"), dict):
                            user_dict = body.get("user")

                        user_obj = None
                        try:
                            if (
                                isinstance(user_dict, dict)
                                and user_dict.get("id") is not None
                            ):
                                user_obj = Users.get_user_by_id(user_dict["id"])
                        except Exception:
                            user_obj = None

                        if user_obj is not None:
                            _run_detached(
                                self.auto_memory_process(
                                    cand, user_obj, __event_emitter__
                                ),
                                name="auto_memory_process_inlet",
                                logger=self.logger,
                            )
                            self.memory_log(
                                "inlet兜底: 已启动异步记忆处理（如版本不触发outlet也能工作）",
                                "info",
                            )
        except Exception as e:
            try:
                self.memory_log("inlet兜底触发失败: %s" % e, "error")
            except Exception:
                pass

        return body

    async def outlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """出口函数 - 添加异步记忆处理"""
        # 兼容不同版本传参：user / __user__，以及部分版本 outlet 不传 user 的情况
        user = (
            user
            if isinstance(user, dict)
            else (__user__ if isinstance(__user__, dict) else None)
        )
        if user is None and isinstance(kwargs.get("user"), dict):
            user = kwargs.get("user")
        if user is None and isinstance(kwargs.get("__user__"), dict):
            user = kwargs.get("__user__")

        user_obj = None
        try:
            if isinstance(user, dict) and user.get("id") is not None:
                user_obj = Users.get_user_by_id(user["id"])
        except Exception:
            user_obj = None

        if user_obj is None:
            user_obj = getattr(self, "current_user_obj", None)

        if user_obj is None:
            return body

        try:
            # user_obj 已准备好

            if user_obj is None:
                self.memory_log("用户对象获取失败", "error")
                return body

            self.current_user_obj = user_obj
        except Exception as e:
            self.memory_log(f"用户信息获取异常: {e}", "error")
            return body

        if self.valves.enable_auto_memory:
            messages = body.get("messages", [])
            if messages and len(messages) >= 2:
                _run_detached(
                    self.auto_memory_process(
                        messages,
                        user_obj,
                        __event_emitter__,
                    ),
                    name="auto_memory_process",
                    logger=self.logger,
                )
                self.memory_log("已启动异步记忆处理", "info")

        # 从模型最终回复中学习“仅文本能力”提示（有些服务不会抛异常，只会文本拒绝图片）
        try:
            msgs = body.get("messages", [])
            last_assistant = None
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    last_assistant = m
                    break
            if last_assistant is not None:
                assistant_text = self.extract_text_from_content(
                    last_assistant.get("content", "")
                )
                if self._is_multimodal_refusal_text(assistant_text):
                    current_model = getattr(self, "_current_model_name", "")
                    self._mark_model_as_text_only(
                        current_model,
                        "模型在回复中声明无法处理图片，已标记为文本模型。",
                    )
                    mk = self._normalize_model_name(current_model)
                    if mk:
                        self.model_multimodal_probe_results[mk] = False
        except Exception:
            pass

        return body
