"""
cost_metrics — 开销统计模块

提供两个核心指标的计算：
1. Total Token Count: 从 agent 输出的 usage 字段读取，或从 trajectory 估算
2. Effective Rounds: 从 trajectory 中统计 assistant 与 tool 的交互轮次

存储位置：meta.cost_metrics
"""

import json
import re
from typing import Any


def compute_effective_rounds(trajectory: list[dict]) -> dict[str, int]:
    """从 OpenAI 格式的 trajectory 计算 Effective Rounds。

    定义：
    - total_rounds: assistant 带 tool_calls 的消息数（每次 LLM→tool 交互 = 1 round）
    - total_assistant_messages: 所有 assistant 消息数（含无 tool_calls 的终止消息）
    - total_tool_calls: 所有 tool 调用总次数（1 条 assistant 可能有多个 tool_calls）
    - total_tool_results: 所有 tool role 消息数
    - total_messages: trajectory 总消息条数
    """
    total_rounds = 0
    total_assistant_messages = 0
    total_tool_calls = 0
    total_tool_results = 0

    for msg in trajectory:
        role = msg.get("role", "")

        if role == "assistant":
            total_assistant_messages += 1
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                total_rounds += 1
                total_tool_calls += len(tool_calls)

        elif role == "tool":
            total_tool_results += 1

    return {
        "effective_rounds": total_rounds,
        "total_assistant_messages": total_assistant_messages,
        "total_tool_calls": total_tool_calls,
        "total_tool_results": total_tool_results,
        "total_messages": len(trajectory),
    }


def estimate_token_count(trajectory: list[dict], model: str = "kimi-k2") -> dict[str, Any]:
    """从 trajectory 内容估算 token 数（当 agent 未提供实际 usage 时的 fallback）。

    使用 char/token 比例估算：
    - 英文: ~4 chars/token
    - 中文: ~2 chars/token
    - 混合: ~3 chars/token（保守估计）

    注意：这只是粗估，实际 token 数取决于模型的 tokenizer。
    """
    total_chars = 0
    input_chars = 0   # user + tool messages (作为 LLM 输入)
    output_chars = 0  # assistant messages (作为 LLM 输出)

    for msg in trajectory:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        chars = len(content)

        # tool_calls 的 arguments 也算 output
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))

        total_chars += chars

        if role == "assistant":
            output_chars += chars
        else:
            input_chars += chars

    # 混合语言按 ~3 chars/token 估算
    chars_per_token = 3.0
    estimated_total = int(total_chars / chars_per_token)
    estimated_input = int(input_chars / chars_per_token)
    estimated_output = int(output_chars / chars_per_token)

    return {
        "estimated_total_tokens": estimated_total,
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "total_chars": total_chars,
        "estimation_method": "char_ratio",
        "chars_per_token": chars_per_token,
    }


# ── Model pricing (USD per 1M tokens) ────────────────────────────────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude Sonnet 4.5
    "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "openai/claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    # Claude Sonnet 4.6
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "openai/claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    # Gemini
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.0},
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.0},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.6},
    # GPT
    "gpt-5.3-chat": {"input": 2.0, "output": 8.0},
    "gpt-5.2": {"input": 2.0, "output": 8.0},
    "gpt-5.2-chat": {"input": 2.0, "output": 8.0},
    "gpt-5": {"input": 2.0, "output": 8.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    # Qwen
    "qwen3.5-plus": {"input": 0.8, "output": 2.0},
    "qwen3-max": {"input": 1.6, "output": 6.4},
    "qwen3-235b-a22b": {"input": 0.8, "output": 2.0},
    # DeepSeek
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek-v3.2": {"input": 0.27, "output": 1.1},
    # Kimi
    "kimi-k2.5": {"input": 0.6, "output": 2.0},
    "kimi-k2-0905-preview": {"input": 0.0, "output": 0.0},
    # Others
    "grok-4": {"input": 3.0, "output": 15.0},
    "glm-5": {"input": 0.5, "output": 2.0},
}


def _match_pricing(model: str) -> dict[str, float] | None:
    """模糊匹配 MODEL_PRICING，支持 model 名包含变体（如 claude-sonnet-4-5-20250514）。"""
    # 精确匹配
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # 模糊匹配：检查 model 是否包含已知 key
    for key, pricing in MODEL_PRICING.items():
        if key in model or model in key:
            return pricing
    # 进一步：检查关键词
    model_lower = model.lower()
    if "claude" in model_lower and "sonnet" in model_lower:
        return MODEL_PRICING.get("anthropic/claude-sonnet-4.5")
    return None


def compute_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> dict[str, Any] | None:
    """计算实际 API 费用（USD）。

    Returns None if model pricing is unknown or zero.
    """
    pricing = _match_pricing(model)
    if not pricing or (pricing["input"] == 0 and pricing["output"] == 0):
        return None

    input_cost = prompt_tokens * pricing["input"] / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000

    return {
        "input": round(input_cost, 6),
        "output": round(output_cost, 6),
        "total": round(input_cost + output_cost, 6),
        "pricing": {
            "input_per_m": pricing["input"],
            "output_per_m": pricing["output"],
            "model": model,
        },
    }


def build_cost_metrics(
    trajectory: list[dict],
    usage: dict[str, Any] | None = None,
    model: str = "",
    time_cost: float = 0.0,
) -> dict[str, Any]:
    """构建完整的 cost_metrics 字典，写入 meta.cost_metrics。

    Args:
        trajectory: OpenAI 格式的 trajectory 消息列表
        usage: agent 输出的实际 token usage（如果有）
        model: 模型名称
        time_cost: 运行耗时（秒）
    """
    rounds = compute_effective_rounds(trajectory)

    metrics: dict[str, Any] = {
        "effective_rounds": rounds["effective_rounds"],
        "rounds_detail": rounds,
        "time_cost": time_cost,
        "model": model,
    }

    # 优先使用实际 usage 数据
    # total_tokens 可能为 0 但 prompt+completion 有值，也视为 actual
    _total = usage.get("total_tokens", 0) if usage else 0
    _prompt = usage.get("prompt_tokens", 0) if usage else 0
    _completion = usage.get("completion_tokens", 0) if usage else 0
    if not _total and (_prompt + _completion) > 0:
        _total = _prompt + _completion
        usage = {**usage, "total_tokens": _total}
    if usage and _total > 0:
        metrics["total_tokens"] = usage["total_tokens"]
        metrics["token_source"] = "actual"
        metrics["usage"] = usage
        # 计算费用
        cost = compute_cost_usd(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=model,
        )
        if cost:
            metrics["cost_usd"] = cost
    else:
        # fallback: 从 trajectory 估算
        est = estimate_token_count(trajectory, model)
        metrics["total_tokens"] = est["estimated_total_tokens"]
        metrics["token_source"] = "estimated"
        metrics["estimation"] = est
        # 估算费用
        cost = compute_cost_usd(
            prompt_tokens=est.get("estimated_input_tokens", 0),
            completion_tokens=est.get("estimated_output_tokens", 0),
            model=model,
        )
        if cost:
            metrics["cost_usd"] = cost

    return metrics


def extract_cost_metrics_from_sample(sample: dict) -> dict[str, Any] | None:
    """从已有的 DB sample 提取或重新计算 cost_metrics。

    用于 backfill 场景：从 trajectories 列和 meta 中恢复 cost_metrics。
    """
    trajectories_str = sample.get("trajectories", "")
    if not trajectories_str:
        return None

    try:
        trajectory = json.loads(trajectories_str)
    except (json.JSONDecodeError, TypeError):
        return None

    meta = sample.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    # 已有实际 usage 数据？
    existing_usage = meta.get("cost_metrics", {}).get("usage")

    return build_cost_metrics(
        trajectory=trajectory,
        usage=existing_usage,
        model=sample.get("model_name", ""),
        time_cost=sample.get("time_cost", 0.0),
    )
