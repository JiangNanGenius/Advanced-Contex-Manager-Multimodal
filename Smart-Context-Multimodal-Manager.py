"""
title: 🚀 Advanced Context Manager - Zero-Loss Coverage-First v2.4.5
author: JiangNanGenius
version: 2.4.7
license: MIT
required_open_webui_version: 0.5.17
Github: https://github.com/JiangNanGenius
description: 性能优化版 - 并发摘要生成 + 缓存机制 + 统计修正
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
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict

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
            # 清理最少使用的项
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
        # 不要deepcopy，直接在原消息上打标签
        self.original_messages = original_messages
        self.order_map = {}  # 消息ID到原始索引的映射
        self.message_ids = {}  # 原始索引到消息ID的映射
        self.content_map = {}  # 内容标识到原始索引的映射

        # ID稳定化：使用可重现的hash，不混入time.time()
        for i, msg in enumerate(self.original_messages):
            content_key = self._generate_stable_content_key(msg)
            # 使用索引+内容生成稳定ID，不包含时间戳
            msg_id = hashlib.md5(f"{i}_{content_key}".encode()).hexdigest()[:16]

            self.order_map[msg_id] = i
            self.message_ids[i] = msg_id
            self.content_map[content_key] = i

            # 在消息中添加顺序标记（原地修改）
            msg["_order_id"] = msg_id
            msg["_original_index"] = i
            msg["_content_key"] = content_key

    def _generate_stable_content_key(self, msg: dict) -> str:
        """生成稳定的消息内容标识（不含时间戳）"""
        role = msg.get("role", "")
        content = msg.get("content", "")

        # 处理多模态内容
        if isinstance(content, list):
            content_parts = []
            for item in content:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", "")[:100])  # 取前100字符
                elif item.get("type") == "image_url":
                    image_data = item.get("image_url", {}).get("url", "")
                    if image_data.startswith("data:"):
                        # 对base64图片，取header+前50字符作为稳定标识
                        try:
                            header, data = image_data.split("base64,", 1)
                            content_parts.append(f"[IMAGE:{header}:{data[:50]}]")
                        except:
                            content_parts.append("[IMAGE:invalid]")
                    else:
                        content_parts.append(f"[IMAGE:url:{image_data[:50]}]")
            content_str = " ".join(content_parts)
        else:
            content_str = str(content)[:200]  # 取前200字符

        return f"{role}:{content_str}"

    def generate_chunk_id(self, msg_id: str, chunk_index: int) -> str:
        """生成chunk ID：{msg_id}#k 格式保持一致"""
        return f"{msg_id}#{chunk_index}"

    def find_current_user_message_index(self, messages: List[dict]) -> int:
        """找到当前用户消息的索引（最后一条用户消息）"""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                return i
        return -1

    def sort_messages_preserve_user(
        self, messages: List[dict], current_user_message: dict = None
    ) -> List[dict]:
        """
        根据原始顺序排序消息，但保护当前用户消息的位置
        确保当前用户消息始终在最后
        """
        if not messages:
            return messages

        # 分离当前用户消息和其他消息
        other_messages = []
        current_user_in_list = None

        for msg in messages:
            if current_user_message and msg.get(
                "_order_id"
            ) == current_user_message.get("_order_id"):
                current_user_in_list = msg
            else:
                other_messages.append(msg)

        # 按原始顺序排序其他消息
        def get_order(msg):
            return msg.get("_original_index", 999999)

        other_messages.sort(key=get_order)

        # 如果找到当前用户消息，将其放在最后
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

        # 最小清理
        content = content.replace("\n", " ").replace("\r", " ")
        content = re.sub(r"\s+", " ", content).strip()
        return content[:100] + "..." if len(content) > 100 else content


class ProcessingStats:
    """处理统计信息记录器 - 增强版"""

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

        # Coverage-First策略统计
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

        # 新增：分块与预算统计
        self.chunked_messages_count = 0  # 被分片的消息数
        self.total_chunks_created = 0  # 创建的总片数
        self.adaptive_blocks_created = 0  # 自适应块数
        self.block_merge_operations = 0  # 块合并操作数
        self.budget_scaling_applied = 0  # 预算缩放应用次数
        self.scaling_factor = 1.0  # 实际缩放因子

        # 护栏统计
        self.guard_a_warnings = 0  # 护栏A警告次数
        self.guard_b_fallbacks = 0  # 护栏B回退次数
        self.id_mapping_errors = 0  # ID映射错误次数

        # 零丢失保障统计
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

        # 其他统计保持不变...
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
        """获取统计摘要 - 简化版本"""
        retention = self.calculate_retention_ratio()
        window_usage = self.calculate_window_usage_ratio()

        return f"""📊 Coverage-First处理完成:
├─ 消息: {self.original_messages} -> {self.final_messages}条 | tokens: {self.original_tokens:,} -> {self.final_tokens:,}
├─ 窗口使用: {window_usage:.1%} | 内容保留: {retention:.1%}
├─ Coverage: 覆盖率{self.coverage_rate:.1%}, 原文{self.coverage_preserved_count}条+摘要{self.coverage_summary_count}条
├─ 性能: 处理{self.processing_time:.1f}s, API调用{self.summary_requests}次, 缓存命中{self.cache_hits}次
└─ 零丢失: {'✅' if self.zero_loss_guarantee else '⚠️'}"""


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
                # 基本清理
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
    """智能模型匹配器 - 更新版本支持更多模型"""

    def __init__(self):
        # 精确匹配规则（更新版本）
        self.exact_matches = {
            # ===== OpenAI / GPT 家族 =====
            "gpt-5": {
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 2000,
            },
            "gpt-5-mini": {
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1800,
            },
            "gpt-5-nano": {
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1600,
            },
            "gpt-4o": {
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            "gpt-4o-mini": {
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1200,
            },
            "gpt-4-turbo": {
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            "gpt-4-vision-preview": {
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            "gpt-4": {
                "family": "gpt",
                "multimodal": False,
                "limit": 8192,
                "image_tokens": 0,
            },
            "gpt-3.5-turbo": {
                "family": "gpt",
                "multimodal": False,
                "limit": 16385,
                "image_tokens": 0,
            },
            "gpt-oss-120b": {
                "family": "gpt",
                "multimodal": False,
                "limit": 124000,
                "image_tokens": 0,
            },
            # ===== Anthropic / Claude 家族（更新 4 / 4.1）=====
            "claude-4.1": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            "claude-4": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            # 兼容保留：旧条目
            "claude-3-5-sonnet": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            "claude-3-opus": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            "claude-3-haiku": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 800,
            },
            "claude-3": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            # thinking 特例
            "anthropic.claude-4-sonnet-latest-extended-thinking": {
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
                "special": "thinking",
            },
            # ===== ByteDance / Doubao 家族（保留原表）=====
            "doubao-1.5-vision-pro": {
                "family": "doubao",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            "doubao-1.5-vision-lite": {
                "family": "doubao",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1200,
            },
            "doubao-1.5-thinking-pro": {
                "family": "doubao",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
                "special": "thinking",
            },
            "doubao-seed-1-6-250615": {
                "family": "doubao",
                "multimodal": False,
                "limit": 256000,
                "image_tokens": 1000,
            },
            "doubao-seed-1-6-251015": {
                "family": "doubao",
                "multimodal": False,
                "limit": 256000,
                "image_tokens": 1000,
            },
            "doubao-seed-1-6-vision-250815": {
                "family": "doubao",
                "multimodal": True,
                "limit": 2560000,
                "image_tokens": 1000,
            },
            "doubao-seed": {
                "family": "doubao",
                "multimodal": False,
                "limit": 50000,
                "image_tokens": 0,
            },
            "doubao": {
                "family": "doubao",
                "multimodal": False,
                "limit": 50000,
                "image_tokens": 0,
            },
            "doubao-1-5-pro-256k": {
                "family": "doubao",
                "multimodal": False,
                "limit": 200000,
                "image_tokens": 0,
            },
            # ===== Google / Gemini（保留原表）=====
            "gemini-pro": {
                "family": "gemini",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "gemini-pro-vision": {
                "family": "gemini",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 800,
            },
            # ===== Alibaba / Qwen =====
            "qwen-vl": {
                "family": "qwen",
                "multimodal": True,
                "limit": 32000,
                "image_tokens": 1000,
            },
            # 新增：Qwen3 Coder Instruct 系列
            "Qwen/Qwen3-Coder-480B-Instruct": {
                "family": "qwen",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "Qwen/Qwen3-Coder-A35B-Instruct": {
                "family": "qwen",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "qwen3-max": {
                "family": "qwen",
                "multimodal": False,
                "limit": 256000,
                "image_tokens": 0,
            },
            "qwen3-vl-plus": {
                "family": "qwen",
                "multimodal": True,
                "limit": 256000,
                "image_tokens": 1000,
            },
            # ===== Huawei / Pangu =====
            "pangu-pro-moe": {
                "family": "pangu",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== Zhipu / GLM =====
            "zai-org/GLM-4.5": {
                "family": "glm",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "glm-4.5": {
                "family": "glm",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "zai-org/GLM-4.6": {
                "family": "glm",
                "multimodal": False,
                "limit": 198000,
                "image_tokens": 0,
            },
            "glm-4.6": {
                "family": "glm",
                "multimodal": False,
                "limit": 198000,
                "image_tokens": 0,
            },
            # ===== Tencent / Hunyuan =====
            "hunyuan-large": {
                "family": "hunyuan",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "hunyuan-pro": {
                "family": "hunyuan",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "hunyuan-vision": {
                "family": "hunyuan",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 800,
            },
            # ===== Moonshot / Kimi =====
            "kimi-k2-instruct": {
                "family": "kimi",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "kimi-k2-thinking": {
                "family": "kimi",
                "multimodal": False,
                "limit": 256000,
                "image_tokens": 0,
            },
            "kimi-k2-turbo-preview": {
                "family": "kimi",
                "multimodal": False,
                "limit": 256000,
                "image_tokens": 0,
            },
            # ===== Baidu / ERNIE 4 / 4.5 =====
            "ernie-4.0": {
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "ernie-4.5": {
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "ernie-4.0-turbo": {
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "ernie-4.5-turbo": {
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== stepfun =====
            "stepfun-ai/step3": {
                "family": "stepfun",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "stepfun-ai/step-3": {
                "family": "stepfun",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "step3": {
                "family": "stepfun",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== DEEPSEEK =====
            "deepseek-chat": {
                "family": "deepseek",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "deepseek-reasoner": {
                "family": "deepseek",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # === inclusionAI ===
            "inclusionAI/Ling-1T": {
                "family": "inclusionai",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            "inclusionAI/Ring-1T": {
                "family": "inclusionai",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # === MiniMax ===
            "MiniMaxAI/MiniMax-M2": {
                "family": "inclusionai",
                "multimodal": False,
                "limit": 192000,
                "image_tokens": 0,
            },
            "MiniMaxAI/MiniMax-M1-80k": {
                "family": "inclusionai",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
        }

        # 模糊匹配规则（更新版本）
        self.fuzzy_patterns = [
            # ===== Thinking 类模型（优先拦截）=====
            {
                "pattern": r".*thinking.*",
                "family": "thinking",
                "multimodal": False,
                "limit": 200000,
                "image_tokens": 0,
                "special": "thinking",
            },
            # ===== GPT-5 / GPT-4 =====
            {
                "pattern": r"gpt-5.*nano.*",
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1600,
            },
            {
                "pattern": r"gpt-5.*mini.*",
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1800,
            },
            {
                "pattern": r"gpt-5(?!.*nano)(?!.*mini).*",
                "family": "gpt",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 2000,
            },
            {
                "pattern": r"gpt-4o.*",
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            {
                "pattern": r"gpt-4.*vision.*",
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            {
                "pattern": r"gpt-4.*turbo.*",
                "family": "gpt",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            {
                "pattern": r"gpt-4.*",
                "family": "gpt",
                "multimodal": False,
                "limit": 8192,
                "image_tokens": 0,
            },
            {
                "pattern": r"gpt-3\.5.*",
                "family": "gpt",
                "multimodal": False,
                "limit": 16385,
                "image_tokens": 0,
            },
            {
                "pattern": r"gpt.*",
                "family": "gpt",
                "multimodal": False,
                "limit": 16385,
                "image_tokens": 0,
            },
            # ===== Claude（4.1 先于 4）=====
            {
                "pattern": r"claude-?4\.1.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"claude-?4(?!\.1).*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            # 兼容旧 3.x（仍保留）
            {
                "pattern": r"claude.*3.*5.*sonnet.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"claude.*3.*opus.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"claude.*3.*haiku.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 800,
            },
            {
                "pattern": r"claude.*3.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"anthropic.*claude.*",
                "family": "claude",
                "multimodal": True,
                "limit": 200000,
                "image_tokens": 1000,
            },
            # ===== Huawei / Pangu =====
            {
                "pattern": r"pangu.*pro.*moe.*",
                "family": "pangu",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== Zhipu / GLM =====
            {
                "pattern": r"(?i)(?:zai-org/)?glm-4\.5(?!V).*",  # 匹配 glm-4.5 (大小写兼容)，排除 glm-4.5V
                "family": "glm",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"(?i)(?:zai-org/)?glm-4\.5V.*",  # 匹配 glm-4.5V (大小写兼容)
                "family": "glm",
                "multimodal": True,
                "limit": 64000,
                "image_tokens": 0,
            },
            # ===== Alibaba / Qwen =====
            {
                "pattern": r"qwen.*/?qwen3.*coder.*480b.*instruct.*",
                "family": "qwen",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"qwen.*/?qwen3.*coder.*a35b.*instruct.*",
                "family": "qwen",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"qwen.*vl.*",
                "family": "qwen",
                "multimodal": True,
                "limit": 32000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"qwen.*",
                "family": "qwen",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== Tencent / Hunyuan =====
            {
                "pattern": r"hunyuan.*vision.*",
                "family": "hunyuan",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 800,
            },
            {
                "pattern": r"hunyuan.*",
                "family": "hunyuan",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== Moonshot / Kimi =====
            {
                "pattern": r"kimi.*k2.*instruct.*",
                "family": "kimi",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"kimi.*k2.*",
                "family": "kimi",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== Baidu / ERNIE =====
            {
                "pattern": r"ernie-?4(?:\.5)?(?:-turbo)?",
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"ernie.*",
                "family": "ernie",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== stepfun =====
            {
                "pattern": r"(?:stepfun(-ai)?/)?step-?3.*",
                "family": "stepfun",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            # ===== ByteDance / Doubao（保留原泛配顺序）=====
            {
                "pattern": r"doubao.*vision.*",
                "family": "doubao",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 1500,
            },
            {
                "pattern": r"doubao.*seed.*",
                "family": "doubao",
                "multimodal": True,
                "limit": 50000,
                "image_tokens": 1000,
            },
            {
                "pattern": r"doubao.*256k.*",
                "family": "doubao",
                "multimodal": False,
                "limit": 200000,
                "image_tokens": 0,
            },
            {
                "pattern": r"doubao.*1\.5.*",
                "family": "doubao",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"doubao.*",
                "family": "doubao",
                "multimodal": False,
                "limit": 50000,
                "image_tokens": 0,
            },
            # ===== Google / Gemini（保留）=====
            {
                "pattern": r"gemini.*vision.*",
                "family": "gemini",
                "multimodal": True,
                "limit": 128000,
                "image_tokens": 800,
            },
            {
                "pattern": r"gemini.*pro.*",
                "family": "gemini",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
            {
                "pattern": r"gemini.*",
                "family": "gemini",
                "multimodal": False,
                "limit": 128000,
                "image_tokens": 0,
            },
        ]

    def match_model(self, model_name: str) -> Dict[str, Any]:
        """智能匹配模型信息"""
        if not model_name:
            return {
                "family": "unknown",
                "multimodal": False,
                "limit": 200000,
                "image_tokens": 1500,
            }

        model_lower = model_name.lower().strip()

        # 精确匹配
        for exact_name, info in self.exact_matches.items():
            if exact_name.lower() == model_lower:
                return {**info, "matched_name": exact_name, "match_type": "exact"}

        # 模糊匹配
        for pattern_info in self.fuzzy_patterns:
            pattern = pattern_info["pattern"]
            if re.match(pattern, model_lower):
                return {
                    "family": pattern_info["family"],
                    "multimodal": pattern_info["multimodal"],
                    "limit": pattern_info["limit"],
                    "image_tokens": pattern_info["image_tokens"],
                    "special": pattern_info.get("special"),
                    "matched_pattern": pattern,
                    "match_type": "fuzzy",
                }

        # 默认匹配
        return {
            "family": "unknown",
            "multimodal": False,
            "limit": 200000,
            "image_tokens": 1500,
            "match_type": "default",
        }


class TokenCalculator:
    """Token计算器 - 优化图片token计算"""

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

        # 简单fallback
        return len(str(text)) // 4

    def calculate_image_tokens(self, image_data: str) -> int:
        """优化的图片token计算 - 根据模型调整"""
        # Ignore the actual image content; use the model's image token estimate.
        if self.model_info:
            return self.model_info.get("image_tokens", 1500)
        return 1500  # 默认fallback


class InputCleaner:
    """输入清洗与严格兜底 - 支持HTTP图片URL"""

    @staticmethod
    def clean_text_for_regex(text: str) -> str:
        """清洗文本用于正则表达式，防止语法错误"""
        if not text:
            return ""

        try:
            # 移除不可见分隔符
            text = text.replace("\u2028", " ").replace("\u2029", " ")
            # 移除其他控制字符
            text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
            # 基本清理
            text = text.replace("\n", " ").replace("\r", " ")
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            print(f"⚠️ 文本清理异常: {str(e)[:100]}")
            # 极端兜底：只保留基本字符
            return "".join(c for c in str(text) if c.isprintable() or c.isspace())[
                :1000
            ]

    @staticmethod
    def validate_and_clean_image_url(image_url: str) -> Tuple[bool, str]:
        """验证并清洗图片URL - 支持HTTP和data URI"""
        if not image_url or not isinstance(image_url, str):
            return False, ""

        try:
            image_url = image_url.strip()

            # 支持HTTP(S) URL
            if image_url.startswith(("http://", "https://")):
                return True, image_url

            # 支持data URI
            if image_url.startswith("data:"):
                if "base64," not in image_url:
                    return False, ""

                header, b64 = image_url.split("base64,", 1)
                # 只接受图片 MIME
                if not header.lower().startswith("data:image/"):
                    return False, ""

                # 去空白
                b64_str = re.sub(r"\s+", "", b64)
                # 太短基本不可能是有效图片
                if len(b64_str) < 100:
                    return False, ""

                # 校验前100字符（补齐 = 避免 padding 错）
                head = b64_str[:100]
                pad_len = (-len(head)) % 4
                try:
                    base64.b64decode(head + ("=" * pad_len), validate=True)
                except Exception:
                    return False, ""

                # 返回清洗后的 data uri
                return True, f"{header}base64,{b64_str}"

            return False, ""
        except Exception as e:
            print(f"⚠️ 图片URL验证异常: {str(e)[:100]}")
            return False, ""

    @staticmethod
    def safe_regex_match(pattern: str, text: str) -> bool:
        """安全的正则匹配，使用search替代match"""
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
                    text_parts.append("[图片]")  # 占位符
            return " ".join(text_parts)
        else:
            return str(content)

    def chunk_single_message(
        self, message: dict, message_order: MessageOrder
    ) -> List[dict]:
        """对单条消息进行分片处理"""
        content_text = self.extract_text_content(message)
        if not self.should_chunk_message(message):
            return [message]  # 不需要分片

        # 分片策略：保持代码块/段落/句子完整
        chunks = self._intelligent_chunk_text(content_text)
        if len(chunks) <= 1:
            return [message]  # 分片后只有一个，直接返回原消息

        # 创建分片消息
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
        """智能文本分片：保持完整性 - 修复换行抹掉问题"""
        if not text:
            return [text]

        # 保留换行用于段落切分：仅去控制字符，并把分隔符正规化成换行
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        text = text.replace("\u2028", "\n").replace("\u2029", "\n")

        # 基本参数
        target_size = self.valves.chunk_target_tokens * 4  # 粗略字符数
        min_size = self.valves.chunk_min_tokens * 4
        max_size = self.valves.chunk_max_tokens * 4
        overlap_size = self.valves.chunk_overlap_tokens * 4

        chunks = []
        current_chunk = ""

        # 按段落分割
        paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 如果当前段落加入后超过目标大小
            if len(current_chunk) + len(paragraph) > target_size and current_chunk:
                # 完成当前chunk
                if len(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                    # 添加重叠内容
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

            # 如果单个段落就超过最大长度，需要句子级切分
            if len(current_chunk) > max_size:
                # 保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

        # 处理最后一个chunk
        if (
            current_chunk and len(current_chunk.strip()) >= min_size // 2
        ):  # 最后一块可以稍短
            chunks.append(current_chunk.strip())
        elif current_chunk and chunks:
            # 太短了，合并到最后一个chunk
            chunks[-1] += "\n\n" + current_chunk.strip()
        elif current_chunk:
            chunks.append(current_chunk.strip())

        # 确保至少有一个chunk
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
    """Coverage计划器 - 重构版，实现自适应分块和一次性缩放"""

    def __init__(self, token_calculator: TokenCalculator, valves):
        self.token_calculator = token_calculator
        self.valves = valves

    def plan_adaptive_coverage_summaries(
        self, scored_msgs: List[dict], total_budget: int
    ) -> Tuple[List[dict], int]:
        """规划自适应覆盖摘要：按原文token量自适应分块 + 一次性比例缩放"""
        if not scored_msgs:
            return [], 0

        # 1. 按分数分档
        HIGH, MID, LOW = self._classify_messages_by_score(scored_msgs)

        # 2. 对低权重消息进行自适应分块
        adaptive_blocks = self._create_adaptive_blocks(LOW)

        # 3. 计算理想预算需求
        entries, ideal_total_cost = self._calculate_ideal_budgets(
            HIGH, MID, adaptive_blocks
        )

        # 4. 一次性比例缩放（如果预算不足）或向上扩张（如果预算充足）
        if ideal_total_cost > total_budget:
            entries, actual_cost = self._apply_proportional_scaling(
                entries, total_budget
            )
        else:
            # 向上扩张模式 - 修正目标使用率
            entries, actual_cost = self._apply_upward_expansion(
                entries, total_budget, ideal_total_cost
            )

        # 5. 极端退化处理
        if actual_cost > total_budget * 1.1:  # 允许10%的容差
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

        # 按原始索引排序
        low_sorted = sorted(low_messages, key=lambda x: x["idx"])

        blocks = []
        current_block = []
        current_tokens = 0
        raw_block_target = self.valves.raw_block_target  # 15k tokens目标

        for item in low_sorted:
            msg_tokens = item["tokens"]

            # 检查是否应该切块
            should_cut_block = False

            # 条件1：累计tokens达到目标
            if current_tokens + msg_tokens > raw_block_target and current_block:
                should_cut_block = True

            # 条件2：时间大跳（相邻消息索引差距大）
            if current_block and abs(item["idx"] - current_block[-1]["idx"]) > 5:
                should_cut_block = True

            # 条件3：角色切换（用户->助手或助手->用户）
            if current_block:
                prev_role = current_block[-1]["msg"].get("role", "")
                curr_role = item["msg"].get("role", "")
                if (
                    prev_role != curr_role
                    and prev_role in ["user", "assistant"]
                    and curr_role in ["user", "assistant"]
                ):
                    should_cut_block = True

            # 条件4：相似度突变（分数差异过大）
            if current_block:
                score_diff = abs(item["score"] - current_block[-1]["score"])
                if score_diff > 0.3:  # 分数差异超过0.3就切块
                    should_cut_block = True

            # 执行切块
            if should_cut_block:
                # 完成当前块
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
                # 开始新块
                current_block = [item]
                current_tokens = msg_tokens
            else:
                # 加入当前块
                current_block.append(item)
                current_tokens += msg_tokens

        # 处理最后一个块
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

        # 块合并：如果块数太多，合并小块
        if len(blocks) > self.valves.max_blocks:
            blocks = self._merge_small_blocks(blocks)

        return blocks

    def _merge_small_blocks(self, blocks: List[dict]) -> List[dict]:
        """合并小块，控制总块数"""
        if len(blocks) <= self.valves.max_blocks:
            return blocks

        # 按raw_tokens排序，优先合并小块
        blocks.sort(key=lambda x: x["raw_tokens"])

        merged_blocks = []
        i = 0
        while i < len(blocks):
            current_block = blocks[i]

            # 尝试与下一个块合并
            if (
                i + 1 < len(blocks)
                and len(merged_blocks) + (len(blocks) - i) > self.valves.max_blocks
            ):
                next_block = blocks[i + 1]

                # 合并条件：总token数不超过2倍目标
                if (
                    current_block["raw_tokens"] + next_block["raw_tokens"]
                    <= self.valves.raw_block_target * 2
                ):
                    # 执行合并
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
                    i += 2  # 跳过两个块
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

        # 高权重和中权重：微摘要
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

        # 自适应块：块预算按大小分配
        for block in adaptive_blocks:
            # 基础预算
            floor_budget = max(
                self.valves.min_block_summary_tokens, self.valves.floor_block
            )
            # 理想预算：基础 + 按原文token量的比例分配
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
        # 计算总的floor和ideal
        total_floors = sum(entry["floor_budget"] for entry in entries)
        total_ideals = sum(entry["ideal_budget"] for entry in entries)

        if total_floors > available_budget:
            # 连floor都超了，执行极端退化
            return self._apply_extreme_fallback_from_entries(entries, available_budget)

        # 计算缩放因子 α = (B - Σfloors) / Σ(ideal_i - floor_i)
        available_for_scaling = available_budget - total_floors
        scalable_amount = total_ideals - total_floors

        if scalable_amount <= 0:
            # 所有条目都是floor，无法缩放
            alpha = 0
        else:
            alpha = available_for_scaling / scalable_amount
            alpha = min(1.0, alpha)  # 限制最大为1.0

        # 应用缩放：budget_i' = floor_i + α * (ideal_i - floor_i)
        total_assigned = 0
        for entry in entries:
            floor_budget = entry["floor_budget"]
            ideal_budget = entry["ideal_budget"]
            scaled_budget = floor_budget + alpha * (ideal_budget - floor_budget)
            entry["budget"] = int(round(scaled_budget))
            total_assigned += entry["budget"]

        # 误差抹平：高分先补，低分先扣
        error = available_budget - total_assigned
        if error != 0:
            # 按分数排序（高分在前）
            scored_entries = [
                (entry.get("score", entry.get("avg_score", 0)), entry)
                for entry in entries
            ]
            scored_entries.sort(key=lambda x: x[0], reverse=True)

            if error > 0:
                # 有余额，高分先补
                for _, entry in scored_entries:
                    if error <= 0:
                        break
                    entry["budget"] += 1
                    error -= 1
            else:
                # 超预算，低分先扣
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
        """向上扩张模式：修正目标使用率与valves一致"""
        expansion_cap = 3.0  # 最大扩张倍数

        # 使用valves中的目标窗口使用率，而不是硬编码0.6
        target_usage = self.valves.target_window_usage  # 0.85

        # 计算扩张因子
        target_cost = int(available_budget * target_usage)
        if ideal_total_cost >= target_cost:
            # 理想成本已经够高，不需要扩张
            for entry in entries:
                entry["budget"] = entry["ideal_budget"]
            return entries, ideal_total_cost

        # 计算扩张倍数
        expansion_factor = min(expansion_cap, target_cost / ideal_total_cost)

        # 优先扩张块摘要和高权重micro
        total_assigned = 0
        for entry in entries:
            base_budget = entry["ideal_budget"]
            if entry["type"] == "adaptive_block":
                # 块摘要优先扩张
                expanded_budget = int(base_budget * expansion_factor)
            elif (
                entry["type"] == "micro"
                and entry.get("score", 0) >= self.valves.coverage_high_score_threshold
            ):
                # 高权重micro适度扩张
                expanded_budget = int(base_budget * min(2.0, expansion_factor))
            else:
                # 其他保持原样
                expanded_budget = base_budget

            entry["budget"] = expanded_budget
            total_assigned += expanded_budget

        # 确保不超预算
        if total_assigned > available_budget:
            # 按比例缩小
            scale_down = available_budget / total_assigned
            for entry in entries:
                entry["budget"] = int(entry["budget"] * scale_down)
            total_assigned = sum(entry["budget"] for entry in entries)

        return entries, total_assigned

    def _apply_extreme_fallback(
        self, scored_msgs: List[dict], available_budget: int
    ) -> Tuple[List[dict], int]:
        """极端退化：单条全局块摘要"""
        # 使用90%的预算作为全局摘要预算，留10%作为缓冲
        global_budget = max(
            self.valves.min_block_summary_tokens, int(available_budget * 0.9)
        )

        # 按索引排序所有消息
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
        # 收集所有消息
        all_msgs = []
        for entry in entries:
            if entry["type"] == "micro":
                all_msgs.append(entry["msg"])
            elif entry["type"] == "adaptive_block":
                all_msgs.extend(entry["msgs"])

        # 按原始索引排序
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
            "avg_score": 0.5,  # 默认分数
        }

        return [entry], global_budget


class Filter:
    class Valves(BaseModel):
        # 基础控制
        enable_processing: bool = Field(
            default=True, description="🔄 启用内容最大化处理"
        )
        excluded_models: str = Field(
            default="", description="🚫 排除模型列表(逗号分隔)"
        )

        # 新增配置：关闭闲时前端、禁用默认 Top-up
        suppress_frontend_when_idle: bool = Field(
            default=True, description="🕶️ 无需处理时不显示任何前端进度/日志"
        )
        enable_window_topup: bool = Field(
            default=False, description="🧯 仅在超限压缩后才允许窗口填充"
        )

        # 核心配置
        max_window_utilization: float = Field(
            default=0.95, description="🪟 最大窗口利用率(95%)"
        )
        aggressive_content_recovery: bool = Field(
            default=True, description="🔄 激进内容合并模式"
        )
        min_preserve_ratio: float = Field(
            default=0.75, description="🔒 最小内容保留比例(75%)"
        )

        # Coverage-First策略配置
        enable_coverage_first: bool = Field(
            default=True, description="🎯 启用Coverage-First策略"
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

        # 新增：自适应分块配置
        raw_block_target: int = Field(
            default=15000, description="🧩 自适应块目标原文tokens"
        )
        floor_block: int = Field(default=300, description="📏 块摘要最小预算tokens")
        max_blocks: int = Field(default=8, description="📚 最大块数量")
        upgrade_min_pct: float = Field(
            default=0.2, description="⬆️ 升级池最小预留比例(20%)"
        )

        # 零丢失保障配置
        enable_zero_loss_guarantee: bool = Field(
            default=True, description="🛡️ 启用零丢失保障"
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
            default=True, description="🚫 禁用保险截断(强制零丢失)"
        )

        # 尽量保留配置
        enable_try_preserve: bool = Field(
            default=True, description="🔒 启用尽量保留机制"
        )
        try_preserve_ratio: float = Field(
            default=0.40, description="🔒 尽量保留预算比例(40%)"
        )
        try_preserve_exchanges: int = Field(
            default=3, description="🔒 尽量保留对话轮次数"
        )

        # 响应空间配置
        response_buffer_ratio: float = Field(
            default=0.06, description="📝 响应空间预留比例(6%)"
        )
        response_buffer_max: int = Field(
            default=3000, description="📝 响应空间最大值(tokens)"
        )
        response_buffer_min: int = Field(
            default=1000, description="📝 响应空间最小值(tokens)"
        )

        # 多模态处理配置
        multimodal_direct_threshold: float = Field(
            default=0.70, description="🎯 多模态直接输入Token预算阈值(70%)"
        )
        preserve_images_in_multimodal: bool = Field(
            default=True, description="📸 多模态模型是否保留原始图片"
        )
        always_process_images_before_summary: bool = Field(
            default=True, description="📝 摘要前总是先处理图片"
        )

        # 上下文最大化处理配置
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

        # 容错机制配置
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

        # 功能开关
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

        # 智能关键字生成和上下文最大化检测
        enable_keyword_generation: bool = Field(
            default=True, description="🔑 启用智能关键字生成"
        )
        enable_ai_context_max_detection: bool = Field(
            default=True, description="🧠 启用AI上下文最大化检测"
        )
        keyword_generation_for_context_max: bool = Field(
            default=True, description="🔑 对上下文最大化启用关键字生成"
        )

        # 统计和调试
        enable_detailed_stats: bool = Field(default=True, description="📊 启用详细统计")
        enable_detailed_progress: bool = Field(
            default=True, description="📱 启用详细进度显示"
        )
        debug_level: int = Field(
            default=0, description="🐛 调试级别 0-3"
        )  # 简化日志：默认为0
        show_frontend_progress: bool = Field(
            default=True, description="📱 显示处理进度"
        )

        # API配置
        api_error_retry_times: int = Field(default=2, description="🔄 API错误重试次数")
        api_error_retry_delay: float = Field(
            default=1.0, description="⏱️ API错误重试延迟(秒)"
        )

        # Token管理
        default_token_limit: int = Field(default=200000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(
            default=0.92, description="🛡️ Token安全比例(92%)"
        )
        target_window_usage: float = Field(
            default=0.85, description="🪟 目标窗口使用率(85%)"
        )
        max_processing_iterations: int = Field(
            default=5, description="🔄 最大处理迭代次数"
        )

        # 保护策略
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

        # 智能分片配置
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

        # 内容优先级设置
        high_priority_content: str = Field(
            default="代码,配置,参数,数据,错误,解决方案,步骤,方法,技术细节,API,函数,类,变量,问题,bug,修复,实现,算法,架构,用户问题,关键回答",
            description="🎯 高优先级内容关键词(逗号分隔)",
        )

        # 统一的API配置
        api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 API基础地址",
        )
        api_key: str = Field(default="", description="🔑 API密钥")

        # 多模态模型配置
        multimodal_model: str = Field(
            default="doubao-1.5-vision-pro-250328", description="🖼️ 多模态模型"
        )

        # 文本模型配置
        text_model: str = Field(
            default="doubao-1-5-lite-32k-250115", description="📝 文本处理模型"
        )

        # 向量模型配置
        text_vector_model: str = Field(
            default="doubao-embedding-large-text-250515", description="🧠 文本向量模型"
        )
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615", description="🧠 多模态向量模型"
        )

        # Vision相关配置
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。特别注意代码、配置、数据等技术信息。保持客观准确，重点突出关键信息。如果图片包含文字内容，请完整转录出来。",
            description="👁️ Vision提示词",
        )
        vision_max_tokens: int = Field(
            default=2500, description="👁️ Vision最大输出tokens"
        )

        # 关键字生成配置
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

        # 上下文最大化检测配置
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

        # 向量检索配置
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

        # 重排序API配置
        rerank_api_base: str = Field(
            default="https://api.bochaai.com", description="🔄 重排序API"
        )
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=100, description="🔝 重排序返回数量")

        # 摘要配置
        max_summary_length: int = Field(default=25000, description="📏 摘要最大长度")
        min_summary_ratio: float = Field(
            default=0.30, description="📏 摘要最小长度比例"
        )
        summary_compression_ratio: float = Field(
            default=0.40, description="📊 摘要压缩比例"
        )
        max_recursion_depth: int = Field(default=3, description="🔄 最大递归深度")

        # 性能配置
        max_concurrent_requests: int = Field(default=6, description="⚡ 最大并发数")
        request_timeout: int = Field(default=90, description="⏱️ 请求超时(秒)")

        # 缓存配置
        enable_embedding_cache: bool = Field(
            default=True, description="💾 启用向量缓存"
        )
        cache_max_size: int = Field(default=1000, description="💾 缓存最大条数")

    def __init__(self):
        print("📍 Coverage-First上下文管理器初始化中...")
        self.valves = self.Valves()

        # 初始化组件
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

        # 解析配置
        self._parse_configurations()

        print("✅ 初始化完成 - Coverage-First上下文管理器")

    def _parse_configurations(self):
        """解析配置项"""
        # 解析高优先级内容关键词
        self.high_priority_keywords = set()
        if self.valves.high_priority_content:
            self.high_priority_keywords = {
                keyword.strip().lower()
                for keyword in self.valves.high_priority_content.split(",")
                if keyword.strip()
            }

    def reset_processing_state(self):
        """重置处理状态"""
        self.current_processing_id = None
        self.message_order = None
        self.current_user_message = None
        self.current_model_info = None
        self.stats = ProcessingStats()

        # 清理缓存
        if self.embedding_cache:
            self.embedding_cache.clear()

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        """分级调试日志 - 简化版本"""
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            # 基本清理
            message = self.input_cleaner.clean_text_for_regex(message)
            print(f"{prefix} {emoji} {message}")

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
        """分析模型信息 - 增加前端显示"""
        model_info = self.model_matcher.match_model(model_name)

        # 设置token计算器的模型信息
        self.token_calculator.set_model_info(model_info)

        # 前端显示模型信息
        multimodal_status = "多模态" if model_info["multimodal"] else "文本"
        family_name = model_info["family"].upper()
        tokens_display = f"{model_info['limit']:,}tokens"
        match_type_display = {"exact": "精确", "fuzzy": "模糊", "default": "默认"}[
            model_info["match_type"]
        ]

        print(f"🎯 模型识别: {model_name}")
        print(f"   ├─ 系列: {family_name}")
        print(f"   ├─ 类型: {multimodal_status}")
        print(f"   ├─ 限制: {tokens_display}")
        print(f"   └─ 匹配: {match_type_display}匹配")

        # 特殊模型提示
        if model_info.get("special") == "thinking":
            print(f"   💭 特殊: Thinking模型")

        # GPT-5系列特别提示
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

        # 处理多模态内容
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
                elif item.get("type") == "image_url":
                    total_tokens += self.token_calculator.calculate_image_tokens("")
        else:
            total_tokens = self.count_tokens(content)

        # 加上角色和格式开销
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
        """移除消息中的内部字段（以下划线开头），确保兼容 OpenAI API。"""
        if not messages:
            return []

        cleaned_messages: List[dict] = []

        for msg in messages:
            if not isinstance(msg, dict):
                cleaned_messages.append(msg)
                continue

            # 去除消息级别的内部字段
            new_msg: dict = {k: v for k, v in msg.items() if not str(k).startswith("_")}

            content = new_msg.get("content")

            # 处理多模态 content 列表
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
        """获取模型的token限制（应用安全系数）"""
        model_info = self.analyze_model(model_name)
        limit = model_info.get("limit", self.valves.default_token_limit)
        safe_limit = int(limit * self.valves.token_safety_ratio)
        self.debug_log(
            2, f"模型token限制: {model_name} -> {limit} -> {safe_limit}", "⚖️"
        )
        return safe_limit

    def is_multimodal_model(self, model_name: str) -> bool:
        """判断模型是否支持多模态输入"""
        model_info = self.analyze_model(model_name)
        return model_info.get("multimodal", False)

    def find_current_user_message(self, messages: List[dict]) -> Optional[dict]:
        """查找当前用户消息（最新的用户输入）"""
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

        # 找到当前用户消息（最新的用户输入）
        current_user_message = None
        current_user_index = -1

        # 从后往前查找最后一条用户消息
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                current_user_message = msg
                current_user_index = i
                break

        if not current_user_message:
            self.debug_log(1, "未找到当前用户消息，所有消息作为历史消息处理", "⚠️")
            return None, messages

        # 分离历史消息（当前用户消息之前的所有消息）
        history_messages = messages[:current_user_index]

        # 更新统计
        self.stats.history_message_separation_count += 1

        self.debug_log(
            1,
            f"消息分离完成: 当前用户消息1条({self.count_message_tokens(current_user_message)}tokens), 历史消息{len(history_messages)}条({self.count_messages_tokens(history_messages):,}tokens)",
            "📋",
        )

        return current_user_message, history_messages

    def calculate_target_tokens(self, model_name: str, current_user_tokens: int) -> int:
        """计算目标token数：模型限制 - 当前用户消息 - 响应空间"""
        model_token_limit = self.get_model_token_limit(model_name)

        # 计算响应空间
        response_buffer = min(
            self.valves.response_buffer_max,
            max(
                self.valves.response_buffer_min,
                int(model_token_limit * self.valves.response_buffer_ratio),
            ),
        )

        # 计算目标：总限制 - 当前用户消息 - 响应缓冲区
        target_tokens = model_token_limit - current_user_tokens - response_buffer

        # 确保不小于基础值
        min_target = max(10000, model_token_limit * 0.3)
        target_tokens = max(target_tokens, min_target)

        self.debug_log(
            1,
            f"目标token计算: {model_token_limit} - {current_user_tokens} - {response_buffer} = {target_tokens}",
            "🎯",
        )

        return int(target_tokens)

    # ========== 新增：正确的"是否需要处理"判定 ==========
    def _needs_processing(
        self, messages: List[dict], model_name: str, target_tokens: int
    ):
        """判定是否需要进行处理"""
        current_tokens = self.count_messages_tokens(messages)
        has_images = self.has_images_in_messages(messages)
        model_is_multimodal = self.is_multimodal_model(model_name)

        token_overflow = current_tokens > target_tokens
        multimodal_incompatible = has_images and (not model_is_multimodal)

        return (
            (token_overflow or multimodal_incompatible),
            token_overflow,
            multimodal_incompatible,
        )

    # ========== 修正的"是否需要最大化"判定 ==========
    def should_force_maximize_content(
        self, messages: List[dict], target_tokens: int
    ) -> bool:
        """判断是否应该强制进行内容最大化处理 - 修正版：只有超限才处理"""
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

        # 使用输入清洗
        cleaned_query = self.input_cleaner.clean_text_for_regex(query_text)

        # 构建提示
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
            # AI检测失败，回退到简单方法
            self.debug_log(1, f"AI检测失败，使用简单方法", "⚠️")
            return self.is_context_max_need_simple(query_text)

    def is_context_max_need_simple(self, query_text: str) -> bool:
        """简单的上下文最大化需求判断（备用方法）"""
        if not query_text:
            return True

        query_text = self.input_cleaner.clean_text_for_regex(query_text)

        # 需要上下文最大化的特征
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

        # 生成缓存键
        content_key = hashlib.md5(text.encode()).hexdigest()[:16]

        # 尝试从缓存获取
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
            # 缓存结果
            if self.embedding_cache:
                self.embedding_cache.set(content_key, embedding)
            self.debug_log(3, f"文本向量获取成功: {len(embedding)}维", "📝")

        return embedding

    async def get_multimodal_embedding_impl(self, content, event_emitter):
        """实际的多模态向量获取实现"""
        client = self.get_api_client()
        if not client:
            return None

        # 处理输入格式
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
                    # 验证并清洗图片数据
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

        # 检查是否为多模态内容
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

    # ========== 相关度计算（并发优化 + 两阶段召回） ==========
    async def compute_relevance_scores(
        self, query_msg: dict, history_msgs: List[dict], progress: ProgressTracker
    ) -> List[dict]:
        """计算所有历史消息的相关度分数（两阶段召回 + 并发优化版本）"""
        if not history_msgs:
            return []

        self.debug_log(
            1, f"开始计算相关度分数: 查询1条，历史{len(history_msgs)}条", "🎯"
        )

        # 获取查询向量
        query_content = query_msg.get("content", "")
        query_text = self.extract_text_from_content(query_content)

        # 第一阶段：轻量级筛选（如果消息数量较多）
        if len(history_msgs) > 40:
            # 轻量级评分先选top-K
            lightweight_scored = self._compute_lightweight_scores(
                query_text, history_msgs
            )
            # 选择top-K进入第二阶段
            top_k = min(self.valves.vector_top_k, len(lightweight_scored))
            lightweight_scored.sort(key=lambda x: x["score"], reverse=True)
            selected_msgs = lightweight_scored[:top_k]

            self.debug_log(
                1,
                f"两阶段召回: {len(history_msgs)} -> {len(selected_msgs)}条进入向量化阶段",
                "⚡",
            )

            # 第二阶段：向量相似度计算
            if len(selected_msgs) > 80:
                # 仍然太多，直接使用轻量级分数
                scored = selected_msgs
            else:
                # 智能向量化策略
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

                # 并发获取向量并重新计算分数
                scored = await self._compute_vector_scores_concurrent(
                    query_vector, selected_msgs, progress
                )
        else:
            # 小数据集：直接使用向量相似度
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

            # 转换为统一格式
            msg_items = []
            for idx, msg in enumerate(history_msgs):
                msg_items.append(
                    {"msg": msg, "idx": idx, "tokens": self.count_message_tokens(msg)}
                )

            scored = await self._compute_vector_scores_concurrent(
                query_vector, msg_items, progress
            )

        self.debug_log(1, f"相关度计算完成: {len(scored)}条消息全部评分", "🎯")

        # 简化Top5打印（只在debug_level >= 2时显示）
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
        """轻量级评分（不使用向量）"""
        scored = []
        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        for idx, msg in enumerate(history_msgs):
            msg_content = msg.get("content", "")
            msg_text = self.extract_text_from_content(msg_content)
            msg_lower = msg_text.lower()
            msg_words = set(msg_lower.split())

            # 基于文本匹配的简单相似度
            common_words = query_words & msg_words
            text_sim = (
                len(common_words) / max(1, len(query_words)) if query_words else 0
            )

            # 额外权重计算
            recency = idx / max(1, len(history_msgs) - 1)
            role = msg.get("role", "")
            role_weight = (
                1.0 if role == "user" else (0.8 if role == "assistant" else 0.6)
            )
            kw_bonus = 1.0 if self.is_high_priority_content(msg_text) else 0.0

            # 综合分数：文本相似度60% + 时间权重20% +角色权重10% + 关键词权重10%
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
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)

        async def get_msg_embedding(item):
            async with semaphore:
                msg = item["msg"]
                msg_content = msg.get("content", "")
                msg_text = self.extract_text_from_content(msg_content)

                # 尝试从缓存获取（基于content_key）
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

                # 缓存结果
                if content_key and msg_vector and self.embedding_cache:
                    self.embedding_cache.set(content_key, msg_vector)

                return item["idx"], msg_vector

        # 并发获取所有向量
        self.stats.concurrent_tasks = len(msg_items)
        embedding_tasks = [get_msg_embedding(item) for item in msg_items]
        embedding_results = await asyncio.gather(
            *embedding_tasks, return_exceptions=True
        )

        # 计算分数
        scored = []
        for item in msg_items:
            # 找到对应的向量结果
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

            # 计算向量相似度
            sim = (
                self.cosine_similarity(query_vector, msg_vector)
                if (query_vector and msg_vector)
                else 0.0
            )

            # 额外权重计算
            recency = item["idx"] / max(1, len(msg_items) - 1)
            role = msg.get("role", "")
            role_weight = (
                1.0 if role == "user" else (0.8 if role == "assistant" else 0.6)
            )
            kw_bonus = 1.0 if self.is_high_priority_content(msg_text) else 0.0

            # 综合分数：向量相似度60% + 时间权重20% + 角色权重10% + 关键词权重10%
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

    # ========== 升级策略（防预算被吃光） ==========
    def select_preserve_upgrades_with_protection(
        self, scored_msgs: List[dict], coverage_entries: List[dict], total_budget: int
    ) -> Tuple[set, int]:
        """选择升级的消息（防预算被吃光版本）"""
        # 先预留升级池
        upgrade_pool = int(total_budget * self.valves.upgrade_min_pct)
        if upgrade_pool <= 0 or not scored_msgs:
            return set(), 0

        self.debug_log(
            1,
            f"升级池保护: 预留{upgrade_pool:,}tokens({self.valves.upgrade_min_pct:.1%})给升级",
            "⬆️",
        )

        # 建立消息ID到摘要成本的映射
        summary_cost_map = defaultdict(int)
        for entry in coverage_entries:
            if entry["type"] == "micro":
                summary_cost_map[entry["msg_id"]] = entry.get(
                    "budget", entry.get("ideal_budget", 0)
                )

        # 构建升级候选列表
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
            # 最近性权重，但设上限防止极长消息挤爆池子
            if item["recency"] > 0.8:
                recency_boost = min(
                    1.2, 1.0 + 0.2 * (2000 / max(upgrade_cost, 1))
                )  # 成本越高，加权越少
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

        # 按价值密度排序
        candidates.sort(key=lambda x: (-x["density"], -x["score"]))

        # 贪心选择，使用升级池预算
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

    # ========== 摘要生成（并发优化） ==========
    async def generate_micro_summary_with_budget_impl(
        self, msg: dict, budget: int, event_emitter
    ):
        """生成单条消息的微摘要（使用缩放后的预算）"""
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
            max_tokens=budget,  # 使用缩放后的预算
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
        """生成自适应块摘要（使用缩放后的预算）"""
        client = self.get_api_client()
        if not client:
            return None

        # 合并消息内容
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
            max_tokens=budget,  # 使用缩放后的预算
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

        # 采样关键消息，避免内容过长
        sampled_msgs = msgs[:: max(1, len(msgs) // 10)]  # 最多采样10条
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
        """并发生成覆盖摘要（使用缩放后的预算）"""
        if not coverage_entries:
            return {}

        self.debug_log(1, f"开始并发生成覆盖摘要: {len(coverage_entries)}个条目", "📝")

        summaries = {}

        # 创建并发任务
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

        # 执行并发任务
        tasks = [generate_single_summary(entry) for entry in coverage_entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        for result in results:
            if isinstance(result, Exception):
                self.stats.api_failures += 1
                continue
            key, summary = result
            if key and summary:
                summaries[key] = summary

        self.debug_log(1, f"并发摘要生成完成: {len(summaries)}个摘要", "📝")
        return summaries

    # ========== 组装阶段双重护栏（修正覆盖率计算） ==========
    async def assemble_coverage_output_with_guards(
        self,
        history_messages: List[dict],
        preserve_set: set,
        coverage_entries: List[dict],
        summaries: Dict[str, str],
        progress: ProgressTracker,
    ) -> List[dict]:
        """组装最终输出（双重护栏版本 - 修正覆盖率计算）"""
        if not history_messages:
            return []

        self.debug_log(1, f"开始组装最终输出: {len(history_messages)}条历史消息", "🔧")

        # 护栏A：校验并日志打印各种条目数量
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

        # 打印前几个未命中micro的msg_id
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

        # 建立映射
        msg_id_to_msg = {
            msg.get("_order_id", f"msg_{i}"): msg
            for i, msg in enumerate(history_messages)
        }

        # 建立块摘要映射
        block_summaries = {}
        block_ranges = {}
        entry_idx_ranges = {}  # 存储每个block_key的idx_range

        for entry in adaptive_block_entries + global_block_entries:
            idx_range = entry["idx_range"]
            block_key = entry.get("block_key", f"block_{idx_range[0]}_{idx_range[1]}")
            entry_idx_ranges[block_key] = idx_range

            if block_key in summaries:
                block_summaries[block_key] = summaries[block_key]
                # 记录这个范围内的所有消息索引
                for idx in range(idx_range[0], idx_range[1] + 1):
                    if idx < len(history_messages):
                        block_ranges[idx] = block_key

        # 计算已被micro/preserve覆盖的索引
        covered_by_micro_or_preserve = set()
        for i, msg in enumerate(history_messages):
            mid = msg.get("_order_id", f"msg_{i}")
            if mid in preserve_set or mid in summaries:  # micro 已生成
                covered_by_micro_or_preserve.add(i)

        final_messages = []
        processed_block_keys = set()
        covered_messages = 0  # 实际被覆盖的消息数

        # 按原始顺序遍历历史消息
        for idx, msg in enumerate(history_messages):
            msg_id = msg.get("_order_id", f"msg_{idx}")
            message_covered = False

            # 检查是否在preserve集合中（升级为原文）
            if msg_id in preserve_set:
                final_messages.append(msg)
                self.stats.coverage_preserved_count += 1
                self.stats.coverage_preserved_tokens += self.count_message_tokens(msg)
                self.debug_log(3, f"使用原文: {msg_id[:8]}", "📄")
                message_covered = True

            # 检查是否有微摘要
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

            # 检查是否在某个块摘要中
            elif idx in block_ranges:
                block_key = block_ranges[idx]
                if (
                    block_key not in processed_block_keys
                    and block_key in block_summaries
                ):
                    # 检查这个块范围里是否还有未被覆盖的索引
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

                        # 标记该块范围内的所有消息为已覆盖
                        for j in range(idx0, idx1 + 1):
                            if j < len(history_messages):
                                message_covered = True
                                break

                # 如果不是第一个或没有未覆盖索引，也认为被块摘要覆盖了
                if idx in block_ranges:
                    message_covered = True

            else:
                # 护栏B：计划micro但最终没落地的条目，回退放简化摘要
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

        # 确保消息顺序正确
        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        # 修正覆盖率计算
        self.stats.coverage_total_messages = len(history_messages)
        self.stats.coverage_rate = covered_messages / max(1, len(history_messages))

        # 护栏A最终验证
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

    # ========== Top-up窗口填充器（修复统计问题） ==========
    def topup_fill_window(
        self,
        final_messages: List[dict],
        scored_msgs: List[dict],
        available_tokens: int,
        summaries: Dict[str, str],
        preserve_set: set,
    ) -> List[dict]:
        """Top-up填充器：把窗口利用率提升到目标值 - 统计修正版"""
        # 一开头就存基线
        initial_tokens = self.count_messages_tokens(final_messages)
        current_tokens = initial_tokens
        target_tokens = int(
            available_tokens * self.valves.target_window_usage
        )  # 使用valves中的设置

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

        # 1) 先把已有 micro 升级为原文（替换掉 micro）
        # 按价值密度（score/tokens）从高到低
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

            # 找到对应的micro摘要消息
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

            # 删除该条 micro，加入原文
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

        # 2) 再从未落地的消息里，按价值密度贪心加入原文
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

        # 3) 重新排序
        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        # 修复统计计算：用基线做差
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

    # ========== Coverage-First主流程（重构版） ==========
    async def process_coverage_first_context_maximization_v2(
        self,
        history_messages: List[dict],
        available_tokens: int,
        progress: ProgressTracker,
        query_message: dict,
        allow_topup: bool = False,  # 新增参数
    ) -> List[dict]:
        """Coverage-First上下文最大化处理主流程 v2.4.5"""
        if not history_messages or not self.valves.enable_coverage_first:
            return history_messages

        await progress.start_phase("Coverage-First处理", len(history_messages))

        self.debug_log(
            1,
            f"Coverage-First开始: {len(history_messages)}条消息, 可用预算: {available_tokens:,}tokens",
            "🎯",
        )

        # Step 0: 消息分片预处理
        if self.valves.enable_smart_chunking:
            await progress.update_progress(0, 8, "消息分片预处理")
            processed_history = self.message_chunker.preprocess_messages_with_chunking(
                history_messages, self.message_order
            )
            self.stats.chunked_messages_count = len(
                [msg for msg in processed_history if msg.get("_is_chunk")]
            )
            # 修正统计计算
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

        # Step 1: 计算相关度分数（两阶段召回 + 并发优化）
        await progress.update_progress(1, 8, "计算相关度分数")
        scored_msgs = await self.compute_relevance_scores(
            query_message, processed_history, progress
        )
        if not scored_msgs:
            self.debug_log(1, "相关度计算失败，使用原始消息", "⚠️")
            return processed_history

        # Step 2: 自适应Coverage规划（按原文token量分块 + 一次性缩放）
        await progress.update_progress(2, 8, "自适应Coverage规划")
        # 为覆盖分配预算（先预留升级池）
        upgrade_pool = int(available_tokens * self.valves.upgrade_min_pct)
        coverage_budget = available_tokens - upgrade_pool
        coverage_entries, coverage_cost = (
            self.coverage_planner.plan_adaptive_coverage_summaries(
                scored_msgs, coverage_budget
            )
        )

        # 记录缩放统计
        if coverage_cost < coverage_budget:
            # 有余额，增加升级池
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

        # Step 3: 升级策略（防预算被吃光）
        await progress.update_progress(3, 8, "升级策略选择")
        preserve_set, upgrade_consumed = self.select_preserve_upgrades_with_protection(
            scored_msgs, coverage_entries, actual_upgrade_pool
        )

        # 更新统计
        self.stats.coverage_upgrade_count = len(preserve_set)
        self.stats.coverage_upgrade_tokens_saved = upgrade_consumed

        # Step 4: 并发生成摘要内容（使用缩放后预算）
        await progress.update_progress(4, 8, "并发生成摘要内容")
        summaries = await self.generate_coverage_summaries_with_budgets(
            coverage_entries, progress
        )

        # Step 5: 双重护栏组装（修正覆盖率计算）
        await progress.update_progress(5, 8, "双重护栏组装")
        final_messages = await self.assemble_coverage_output_with_guards(
            processed_history, preserve_set, coverage_entries, summaries, progress
        )

        # Step 6: Top-up窗口填充（修正统计）- 只在允许时执行
        if allow_topup and self.valves.enable_window_topup:
            await progress.update_progress(6, 8, "Top-up窗口填充")
            final_messages = self.topup_fill_window(
                final_messages, scored_msgs, available_tokens, summaries, preserve_set
            )

        # Step 7: 最终统计
        await progress.update_progress(7, 8, "最终统计计算")
        final_tokens = self.count_messages_tokens(final_messages)
        self.stats.coverage_budget_usage = (
            final_tokens / available_tokens if available_tokens > 0 else 0
        )

        # 确保消息顺序正确
        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )

        await progress.update_progress(8, 8, "处理完成")

        self.debug_log(
            1,
            f"Coverage-First完成: {len(processed_history)} -> {len(final_messages)}条消息",
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

        # 使用InputCleaner验证（支持HTTP和data URI）
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
        self, message: dict, progress: ProgressTracker
    ) -> dict:
        """处理单条消息中的图片"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message

        # 检查是否包含图片
        images = [item for item in content if item.get("type") == "image_url"]
        if not images:
            return message

        self.debug_log(2, f"处理消息中的图片: {len(images)}张", "🖼️")

        # 处理图片
        processed_content = []
        image_count = 0

        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text.strip():
                    processed_content.append(text)
            elif item.get("type") == "image_url":
                image_count += 1
                image_data = item.get("image_url", {}).get("url", "")

                # 校验 URL / data URI
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

                # describe_image 内部也会再次校验，这里传 cleaned 或原始都可
                description = await self.describe_image(
                    cleaned, progress.event_emitter if progress else None
                )

                image_description = f"[图片{image_count}描述] {description}"
                processed_content.append(image_description)

        # 创建新消息
        processed_message = copy.deepcopy(message)
        processed_message["content"] = (
            "\n".join(processed_content) if processed_content else ""
        )
        processed_message["_images_processed"] = image_count

        # 更新统计
        self.stats.multimodal_processed += image_count

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
        # 检查是否包含图片
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return "text_only", "无图片内容，按文本处理"

        # 判断模型类型
        is_multimodal = self.is_multimodal_model(model_name)
        self.debug_log(1, f"模型分析: {model_name} | 多模态支持: {is_multimodal}", "🤖")

        if is_multimodal:
            budget_sufficient = self.calculate_multimodal_budget_sufficient(
                messages, target_tokens
            )
            if budget_sufficient:
                return "direct_multimodal", "多模态模型，Token预算充足，直接输入"
            else:
                return "multimodal_rag", "多模态模型，Token预算不足，使用多模态向量RAG"
        else:
            return "vision_to_text", "纯文本模型，先识别图片再处理"

    async def process_multimodal_content(
        self,
        messages: List[dict],
        model_name: str,
        target_tokens: int,
        progress: ProgressTracker,
    ) -> List[dict]:
        """多模态内容处理"""
        if not self.valves.enable_multimodal:
            return messages

        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages

        # 确定策略
        strategy, strategy_desc = await self.determine_multimodal_processing_strategy(
            messages, model_name, target_tokens
        )

        self.debug_log(1, f"多模态策略: {strategy} - {strategy_desc}", "🎯")

        if strategy == "text_only":
            return messages
        elif strategy == "direct_multimodal":
            return messages
        elif strategy == "vision_to_text":
            await progress.start_phase("视觉识别转文本", len(messages))

            # 并发处理
            semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)

            async def process_single_message(i, message):
                if self.has_images_in_content(message.get("content")):
                    async with semaphore:
                        processed_message = await self.process_message_images(
                            message, progress
                        )
                        return processed_message
                else:
                    return message

            process_tasks = []
            for i, message in enumerate(messages):
                task = process_single_message(i, message)
                process_tasks.append(task)

            processed_messages = await asyncio.gather(*process_tasks)

            if self.message_order:
                processed_messages = self.message_order.sort_messages_preserve_user(
                    processed_messages, self.current_user_message
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

        # 按优先级排序
        if preserve_priority:
            message_priorities = []
            for i, msg in enumerate(messages):
                priority_score = self._calculate_message_priority(msg, i, len(messages))
                message_priorities.append((i, msg, priority_score))

            message_priorities.sort(key=lambda x: x[2], reverse=True)
        else:
            message_priorities = [(i, msg, 1.0) for i, msg in enumerate(messages)]

        # 智能选择
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

        # 填补空隙
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

        # 按原始索引排序
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
        """计算消息优先级分数 - 增加长度惩罚"""
        priority = 1.0

        # 角色优先级
        role = msg.get("role", "")
        if role == "user":
            priority += 2.0
        elif role == "assistant":
            priority += 1.5
        elif role == "system":
            priority += 3.0

        # 位置优先级
        position_score = index / total_count if total_count > 0 else 0
        priority += position_score * 2.0

        # 内容优先级
        content_text = self.extract_text_from_content(msg.get("content", ""))
        if self.is_high_priority_content(content_text):
            priority += 1.5

        # 长度因子 - 增强长度惩罚
        content_length = len(content_text)
        if 100 < content_length < 2000:
            priority += 0.5
        elif content_length > 5000:
            priority -= 1.0  # 增加惩罚
        elif content_length > 10000:
            priority -= 2.0  # 极长文本重度惩罚

        # 多模态内容优先级
        if self.has_images_in_content(msg.get("content")):
            priority += 1.0

        # 摘要消息的优先级
        if msg.get("_is_summary"):
            priority += 0.8

        # 分片消息的优先级
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

        # 检查当前用户消息是否在最后位置
        if final_messages and final_messages[-1].get("role") == "user":
            current_id = self.current_user_message.get("_order_id")
            last_id = final_messages[-1].get("_order_id")
            if current_id == last_id:
                return final_messages

        # 修复位置
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
        """内容最大化综合处理 v2.4.5"""
        start_time = time.time()

        # 获取真实模型限制
        current_model_name = getattr(self, "_current_model_name", "unknown")
        if hasattr(self, "current_model_info") and self.current_model_info:
            model_limit = self.current_model_info.get(
                "limit", self.valves.default_token_limit
            )
            safe_limit = int(model_limit * self.valves.token_safety_ratio)
        else:
            safe_limit = self.get_model_token_limit(current_model_name)

        # 初始化统计
        self.stats.original_tokens = self.count_messages_tokens(messages)
        self.stats.original_messages = len(messages)
        self.stats.token_limit = safe_limit
        self.stats.target_tokens = target_tokens
        current_tokens = self.stats.original_tokens

        self.debug_log(
            1,
            f"Coverage-First处理开始: {current_tokens:,} tokens, {len(messages)} 条消息",
            "🎯",
        )

        await progress.start_phase("Coverage-First处理", 10)

        # 1. 消息分离
        await progress.update_progress(1, 10, "分离当前用户消息和历史消息")
        current_user_message, history_messages = (
            self.separate_current_and_history_messages(messages)
        )
        self.current_user_message = current_user_message

        # 系统消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]

        if current_user_message:
            self.stats.current_user_tokens = self.count_message_tokens(
                current_user_message
            )

        # 2. 检测上下文最大化需求
        need_context_max = False
        if current_user_message and self.valves.enable_context_maximization:
            query_text = self.extract_text_from_content(
                current_user_message.get("content", "")
            )
            need_context_max = await self.detect_context_max_need(
                query_text, progress.event_emitter
            )

            if need_context_max:
                self.debug_log(
                    1, f"检测到需要上下文最大化，启用Coverage-First策略", "📚"
                )

        # 3. 计算保护消息的token
        protected_messages = system_messages[:]
        protected_tokens = self.count_messages_tokens(protected_messages)
        available_for_processing = (
            target_tokens - protected_tokens - self.stats.current_user_tokens
        )

        self.debug_log(
            1, f"历史消息可用处理空间: {available_for_processing:,}tokens", "💰"
        )

        # 4. 处理历史消息
        if not history_messages:
            final_messages = system_messages[:]
            if current_user_message:
                final_messages.append(current_user_message)
            await progress.complete_phase("无历史消息需要处理")
            return final_messages

        # 5. 使用Coverage-First v2.4.5策略
        if (
            need_context_max
            and self.valves.enable_context_maximization
            and self.valves.enable_coverage_first
        ):
            await progress.update_progress(2, 10, "Coverage-First专用处理")
            processed_history = (
                await self.process_coverage_first_context_maximization_v2(
                    history_messages,
                    available_for_processing,
                    progress,
                    current_user_message,
                    allow_topup=True and self.valves.enable_window_topup,  # 修改调用
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

        # 6. 零丢失保障检查
        await progress.update_progress(6, 10, "零丢失保障检查")
        final_history = processed_history
        final_tokens = self.count_messages_tokens(final_history)

        if (
            final_tokens > available_for_processing
            and self.valves.disable_insurance_truncation
        ):
            self.debug_log(1, f"预算超限但禁用截断，保证零丢失", "🛡️")
            self.stats.insurance_truncation_avoided += 1
        elif final_tokens > available_for_processing:
            self.debug_log(1, f"超出预算，启用保险截断", "✂️")
            final_history = self.smart_truncate_messages(
                final_history, available_for_processing, True
            )
            final_tokens = self.count_messages_tokens(final_history)
            self.stats.zero_loss_guarantee = False

        # 7. 组合最终结果
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

        # 8. 用户消息保护
        await progress.update_progress(9, 10, "用户消息保护验证")
        final_messages = self.ensure_current_user_message_preserved(final_messages)

        # 9. 更新统计
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
            f"Coverage-First处理完成: 保留{retention_ratio:.1%} 窗口使用{window_usage:.1%} 零丢失{'保障成功' if self.stats.zero_loss_guarantee else '部分失效'}",
            "🎯",
        )

        await progress.complete_phase(
            f"覆盖率{self.stats.coverage_rate:.1%} 预算使用{window_usage:.1%} "
            f"零丢失{'成功' if self.stats.zero_loss_guarantee else '失效'} "
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

    # ========== 主要入口函数 ==========
    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """入口函数 v2.4.5 - 性能优化版本"""
        if self.valves.debug_level >= 1:
            print("🚀 Coverage-First上下文管理器启动")

        if not self.valves.enable_processing:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        model_name = body.get("model", "未知")
        if self.is_model_excluded(model_name):
            return body

        # 重置处理状态
        self.reset_processing_state()

        # 保存当前模型名
        self._current_model_name = model_name

        # 分析模型信息
        self.current_model_info = self.analyze_model(model_name)

        # 计算目标/判定是否需要处理
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

        # 修正的进度追踪器逻辑
        show_progress = needs_proc or not self.valves.suppress_frontend_when_idle
        progress = ProgressTracker(__event_emitter__ if show_progress else None)

        # 初始化消息顺序管理器（不再deepcopy，直接在原消息上打标签）
        self.message_order = MessageOrder(messages)

        # 【关键修复】：使用带_order_id的消息列表
        messages = self.message_order.original_messages

        # 消息分离（使用带ID的消息）
        current_user_message, history_messages = (
            self.separate_current_and_history_messages(messages)
        )
        self.current_user_message = current_user_message

        # 更新统计
        self.stats.token_limit = model_token_limit
        self.stats.target_tokens = target_tokens
        self.stats.current_user_tokens = current_user_tokens

        if self.valves.debug_level >= 1:
            print(
                f"模型: {self.current_model_info['family']} | tokens: {original_tokens:,}/{model_token_limit:,} | 历史: {len(history_messages)}条"
            )

        # 生成处理ID
        if current_user_message:
            content_preview = self.message_order.get_message_preview(
                current_user_message
            )
            processing_id = hashlib.md5(
                f"{current_user_message.get('_order_id', '')}{content_preview}{time.time()}".encode()
            ).hexdigest()[:8]
            self.current_processing_id = processing_id

            # AI"上下文最大化"只作为策略参考，绝不强制触发处理
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

        # 最终是否进入重处理：仅当确实"需要处理"
        should_maximize = needs_proc

        try:
            # 1. 多模态处理
            if self.valves.enable_detailed_progress:
                await progress.start_phase("多模态处理", 1)

            processed_messages = await self.process_multimodal_content(
                messages, model_name, target_tokens, progress
            )
            processed_tokens = self.count_messages_tokens(processed_messages)

            # 轻量兜底：若用户消息仍是列表/字典，强制转为纯文本（保留未描述的图片为占位符）
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

            # 🔁 基于“转写后”重新评估是否需要处理（权威判定）
            _post_user = self.find_current_user_message(processed_messages)
            _post_user_tokens = (
                self.count_message_tokens(_post_user) if _post_user else 0
            )
            target_tokens = self.calculate_target_tokens(model_name, _post_user_tokens)
            post_needs_proc, post_token_overflow, post_mm_incompat = (
                self._needs_processing(processed_messages, model_name, target_tokens)
            )

            # 2) 若转写后已满足窗口/多模态约束 => 直接原文通行
            if not post_needs_proc:
                self.stats.original_tokens = self.count_messages_tokens(messages)
                self.stats.original_messages = len(messages)
                self.stats.final_tokens = processed_tokens
                self.stats.final_messages = len(processed_messages)
                body["messages"] = copy.deepcopy(processed_messages)
                # 确保移除内部字段，兼容 OpenAI API
                body["messages"] = self.strip_internal_fields(body["messages"])
                if self.valves.debug_level >= 1:
                    print("无需处理：多模态转写后未超限，直接返回原文（或转写后）")
                return body

            # 2.5) （后移）AI 上下文最大化检测：仅当仍需处理
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

            # 3. Coverage-First 仅在确需处理时启用
            if should_maximize:
                final_messages = (
                    await self.maximize_content_comprehensive_processing_v2(
                        processed_messages, target_tokens, progress
                    )
                )

                # 打印详细统计
                self.print_detailed_stats()

                body["messages"] = copy.deepcopy(final_messages)

                # 最终统计
                final_tokens = self.count_messages_tokens(final_messages)
                window_utilization = (
                    final_tokens / target_tokens if target_tokens > 0 else 0
                )

                if self.valves.debug_level >= 1:
                    print(
                        f"处理完成: {len(final_messages)}条消息, {final_tokens:,}tokens, 利用率{window_utilization:.1%}, 零丢失{'✅' if self.stats.zero_loss_guarantee else '⚠️'}"
                    )

                # 验证当前用户消息保护
                if current_user_message and final_messages:
                    last_msg = final_messages[-1]
                    if last_msg.get("role") == "user" and self.valves.debug_level >= 1:
                        print(f"当前用户消息保护: ✅")
            else:
                # 直接使用处理后的消息
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
            print("🏁 Coverage-First处理完成")

        # 出口统一清理内部字段，确保兼容 OpenAI API
        if isinstance(body.get("messages"), list):
            body["messages"] = self.strip_internal_fields(body["messages"])

        return body

    async def outlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """出口函数"""
        return body
