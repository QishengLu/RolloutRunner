"""
usage_tracker — 通用 token 用量追踪器

各 agent_runner.py 导入此模块，自动拦截所有 OpenAI-compatible API 调用并累加 token 用量。
最终通过 get_usage() 获取汇总数据，写入 agent 输出的 usage 字段。

核心方式：install_openai_hooks() — monkey-patch OpenAI SDK 的 create 方法
这是最通用的方式，无论上层使用 LangChain、litellm、MetaChain 还是直调 OpenAI，
都会经过 openai.resources.chat.completions.Completions.create，因此可以一次性拦截所有调用。

用法（推荐，在 agent_runner.py 顶部导入即可）：

    import sys
    sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
    from src.usage_tracker import UsageTracker

    tracker = UsageTracker()
    tracker.install_openai_hooks()

    # ... 正常运行 agent ...

    # 输出时附带 usage
    result = {"output": ..., "trajectory": ..., "usage": tracker.get_usage()}
    print(json.dumps(result, ensure_ascii=False))
"""

import sys
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageTracker:
    """线程安全的 token 用量累加器。"""

    _total_tokens: int = 0
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _reasoning_tokens: int = 0
    _call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _hooked: bool = False

    def track(self, response: Any) -> None:
        """从 API response 对象提取 usage 并累加。

        兼容 OpenAI SDK response 和 litellm response：
          response.usage.prompt_tokens
          response.usage.completion_tokens
          response.usage.total_tokens

        也兼容 dict 格式的 response（如 {"usage": {"total_tokens": N}}）。
        """
        usage = None

        # OpenAI/litellm SDK response object
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
        # dict response
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]

        if usage is None:
            return

        with self._lock:
            self._call_count += 1

            if hasattr(usage, "input_tokens"):
                # Anthropic SDK format: input_tokens / output_tokens (no total_tokens field)
                input_t = getattr(usage, "input_tokens", 0) or 0
                output_t = getattr(usage, "output_tokens", 0) or 0
                self._prompt_tokens += input_t
                self._completion_tokens += output_t
                self._total_tokens += input_t + output_t
            elif hasattr(usage, "total_tokens"):
                # OpenAI SDK format: total_tokens / prompt_tokens / completion_tokens
                self._total_tokens += usage.total_tokens or 0
                self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                # reasoning_tokens 在某些模型（o3等）中可用
                reasoning = getattr(usage, "reasoning_tokens", 0) or 0
                if not reasoning:
                    detail = getattr(usage, "completion_tokens_details", None)
                    if detail:
                        reasoning = getattr(detail, "reasoning_tokens", 0) or 0
                self._reasoning_tokens += reasoning
            elif isinstance(usage, dict):
                if "input_tokens" in usage:
                    # Anthropic dict format
                    input_t = usage.get("input_tokens", 0) or 0
                    output_t = usage.get("output_tokens", 0) or 0
                    self._prompt_tokens += input_t
                    self._completion_tokens += output_t
                    self._total_tokens += input_t + output_t
                else:
                    # OpenAI dict format
                    self._total_tokens += usage.get("total_tokens", 0) or 0
                    self._prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self._completion_tokens += usage.get("completion_tokens", 0) or 0
                    self._reasoning_tokens += usage.get("reasoning_tokens", 0) or 0

    def track_manual(
        self,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """手动添加 token 用量（框架不直接暴露 usage 时使用）。"""
        with self._lock:
            self._call_count += 1
            self._total_tokens += total_tokens
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._reasoning_tokens += reasoning_tokens

    def get_usage(self) -> dict[str, int]:
        """获取汇总的 token 用量。"""
        with self._lock:
            return {
                "total_tokens": self._total_tokens,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "reasoning_tokens": self._reasoning_tokens,
                "llm_call_count": self._call_count,
            }

    # ── Normalize to OpenAI string format ──────────────────────────────────────

    @staticmethod
    def _normalize_to_openai_format(kwargs: dict) -> dict:
        """确保消息为纯 OpenAI 格式（string content），兼容 shubiaobiao proxy。

        shubiaobiao proxy 将 Claude 模型路由到 AWS Bedrock，proxy 自身负责
        OpenAI→Bedrock 格式转换。但 proxy 要求输入为标准 OpenAI 格式：
        - content 为 string（不是 list of blocks）
        - tool_calls 保留在 assistant 消息上
        - role="tool" 保留

        某些框架（如 LangChain）会发送 content 为 list 格式（如
        [{"type": "text", "text": "..."}]），proxy 无法正确处理。
        此方法将 list content 转回 string，让 proxy 正确工作。
        """
        messages = kwargs.get("messages")
        if not messages:
            return kwargs

        new_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue

            content = msg.get("content")

            # Convert list content → string (extract text from content blocks)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                # Only convert if all blocks are text (no images etc.)
                if len(text_parts) == len(content):
                    msg = {**msg, "content": "\n".join(text_parts) if text_parts else ""}
                # Otherwise keep as-is (multimodal content)

            new_messages.append(msg)

        kwargs = {**kwargs, "messages": new_messages}

        # Claude models via Bedrock/shubiaobiao proxy reject requests that specify
        # both temperature and top_p simultaneously. Strip top_p when temperature
        # is already present (affects TaskWeaver which sends both as 0 by default).
        if "temperature" in kwargs and "top_p" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "top_p"}

        return kwargs

    # ── Universal OpenAI SDK hook ────────────────────────────────────────────

    def install_openai_hooks(self) -> None:
        """Monkey-patch OpenAI SDK 的 create 方法，自动拦截所有 LLM 调用。

        无论上层框架是 LangChain、litellm、MetaChain 还是直调 OpenAI，
        最终都通过 openai.resources.chat.completions 的 create/acreate 调用 API。
        此方法在返回结果后自动提取 usage 并累加。

        额外功能：自动将 content 转为 list 格式，兼容 AWS Bedrock 后端。

        安全性：
        - 只 hook 一次（幂等）
        - 保留原始方法引用，不影响返回值
        - 异常不影响原调用
        """
        if self._hooked:
            return

        try:
            import openai.resources.chat.completions as completions_mod
        except ImportError:
            print("[UsageTracker] openai not installed, skipping hooks", file=sys.stderr)
            return

        tracker = self

        # ── Sync hook ────────────────────────────────────────────────────
        _orig_create = completions_mod.Completions.create

        def _hooked_create(self_inner, *args, **kwargs):
            kwargs = tracker._normalize_to_openai_format(kwargs)
            is_stream = kwargs.get("stream", False)
            response = _orig_create(self_inner, *args, **kwargs)
            if is_stream:
                # Streaming calls: return the original Stream object unchanged.
                # Wrapping as a generator or injecting stream_options breaks frameworks
                # like TaskWeaver that rely on the Stream object's type/interface.
                # Streaming token counts will fall back to "estimated" in cost_metrics.
                return response
            try:
                # When called via with_raw_response, the SDK returns a
                # LegacyAPIResponse (no .usage attr) instead of ChatCompletion.
                # We .parse() it to extract the ChatCompletion for tracking.
                if hasattr(response, "parse") and not hasattr(response, "usage"):
                    tracker.track(response.parse())
                else:
                    tracker.track(response)
            except Exception:
                pass
            return response

        completions_mod.Completions.create = _hooked_create

        # ── Async hook ───────────────────────────────────────────────────
        _orig_acreate = completions_mod.AsyncCompletions.create

        async def _hooked_acreate(self_inner, *args, **kwargs):
            kwargs = tracker._normalize_to_openai_format(kwargs)
            is_stream = kwargs.get("stream", False)
            response = await _orig_acreate(self_inner, *args, **kwargs)
            if is_stream:
                # Same as sync: return original async Stream unchanged.
                return response
            try:
                if hasattr(response, "parse") and not hasattr(response, "usage"):
                    tracker.track(response.parse())
                else:
                    tracker.track(response)
            except Exception:
                pass
            return response

        completions_mod.AsyncCompletions.create = _hooked_acreate

        self._hooked = True
        print("[UsageTracker] OpenAI SDK hooks installed", file=sys.stderr)

    # ── litellm hook ──────────────────────────────────────────────────────

    _litellm_hooked: bool = False

    def install_litellm_hooks(self) -> None:
        """Monkey-patch litellm.completion / litellm.acompletion。

        litellm 用自己的 httpx 客户端，绕过 OpenAI SDK，所以需要单独 hook。
        litellm response 对象有 .usage 属性，格式与 OpenAI SDK 兼容。
        """
        if self._litellm_hooked:
            return

        try:
            import litellm
        except ImportError:
            print("[UsageTracker] litellm not installed, skipping hooks", file=sys.stderr)
            return

        tracker = self

        # ── Sync hook ────────────────────────────────────────────────────
        _orig_completion = litellm.completion

        def _hooked_completion(*args, **kwargs):
            response = _orig_completion(*args, **kwargs)
            try:
                tracker.track(response)
            except Exception:
                pass
            return response

        litellm.completion = _hooked_completion

        # ── Async hook ───────────────────────────────────────────────────
        _orig_acompletion = litellm.acompletion

        async def _hooked_acompletion(*args, **kwargs):
            response = await _orig_acompletion(*args, **kwargs)
            try:
                tracker.track(response)
            except Exception:
                pass
            return response

        litellm.acompletion = _hooked_acompletion

        self._litellm_hooked = True
        print("[UsageTracker] litellm hooks installed", file=sys.stderr)

    # ── Anthropic SDK hook ────────────────────────────────────────────────

    _anthropic_hooked: bool = False

    def install_anthropic_hooks(self) -> None:
        """Monkey-patch Anthropic SDK Messages.create (sync + async).

        LangChain ChatAnthropic 内部调用 anthropic.Anthropic().messages.create()，
        绕过 OpenAI SDK。此方法直接 hook Anthropic SDK，无需改变 agent 框架代码。

        Anthropic response 格式：usage.input_tokens / usage.output_tokens
        track() 已支持此格式，会自动映射到 prompt_tokens / completion_tokens。
        """
        if self._anthropic_hooked:
            return

        try:
            from anthropic.resources.messages.messages import Messages, AsyncMessages
        except ImportError:
            print("[UsageTracker] anthropic not installed, skipping hooks", file=sys.stderr)
            return

        tracker = self

        # ── Sync hook ────────────────────────────────────────────────────
        _orig_create = Messages.create

        def _hooked_anthropic_create(self_inner, *args, **kwargs):
            response = _orig_create(self_inner, *args, **kwargs)
            try:
                # Pass the full response so track() can extract .usage internally
                tracker.track(response)
            except Exception:
                pass
            return response

        Messages.create = _hooked_anthropic_create

        # ── Async hook ───────────────────────────────────────────────────
        _orig_acreate = AsyncMessages.create

        async def _hooked_anthropic_acreate(self_inner, *args, **kwargs):
            response = await _orig_acreate(self_inner, *args, **kwargs)
            try:
                tracker.track(response)
            except Exception:
                pass
            return response

        AsyncMessages.create = _hooked_anthropic_acreate

        self._anthropic_hooked = True
        print("[UsageTracker] Anthropic SDK hooks installed", file=sys.stderr)

    # ── LangChain callback (alternative) ─────────────────────────────────

    def langchain_callback(self) -> Any:
        """返回一个 LangChain CallbackHandler，自动追踪每次 LLM 调用的 token 用量。

        用法：
            tracker = UsageTracker()
            cb = tracker.langchain_callback()
            model.invoke(messages, config={"callbacks": [cb]})
        """
        tracker = self

        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            return None

        class _LCTokenCallback(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                llm_output = getattr(response, "llm_output", None) or {}
                token_usage = llm_output.get("token_usage", {})
                if token_usage:
                    tracker.track_manual(
                        total_tokens=token_usage.get("total_tokens", 0),
                        prompt_tokens=token_usage.get("prompt_tokens", 0),
                        completion_tokens=token_usage.get("completion_tokens", 0),
                    )

        return _LCTokenCallback()
