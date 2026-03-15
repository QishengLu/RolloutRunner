# LangChain ChatAnthropic Usage Tracking Fix

## Problem

Agents using `langchain_anthropic.ChatAnthropic` (thinkdepthai, deerflow, aiq) showed `token_source: "estimated"` even after installing `UsageTracker.install_openai_hooks()`.

**Root cause**: `ChatAnthropic` bypasses the OpenAI SDK entirely. It calls `anthropic.resources.messages.messages.Messages.create` directly via the Anthropic SDK. The OpenAI monkey-patch never intercepts these calls.

**Impact**: Estimated tokens = trajectory char count ÷ 3 (O(n) sum of messages). Actual tokens follow O(n²) growth because each LLM call sends the entire accumulated context as input. For a 21-round conversation, this caused a **12.5× underestimate** (36,842 estimated vs 462,263 actual).

## Solution

Call `_tracker.install_anthropic_hooks()` in addition to `install_openai_hooks()`.

### agent_runner.py (top of file, before all other imports)

```python
import sys

sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
from src.usage_tracker import UsageTracker

_tracker = UsageTracker()
_tracker.install_openai_hooks()
_tracker.install_anthropic_hooks()  # ChatAnthropic 走 Anthropic SDK，需要单独 hook

# 清理 RolloutRunner 路径（避免与 agent 自己的 src 包冲突）
sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
for mod_name in list(sys.modules):
    if mod_name == "src" or mod_name.startswith("src."):
        del sys.modules[mod_name]
```

### Output (end of run function)

```python
result = {
    "output": ...,
    "trajectory": ...,
    "usage": _tracker.get_usage(),
}
```

## How install_anthropic_hooks() Works

```python
def install_anthropic_hooks(self) -> None:
    from anthropic.resources.messages.messages import Messages, AsyncMessages
    tracker = self
    _orig_create = Messages.create

    def _hooked_anthropic_create(self_inner, *args, **kwargs):
        response = _orig_create(self_inner, *args, **kwargs)
        try:
            tracker.track(response)  # Pass full response, NOT response.usage
        except Exception:
            pass
        return response

    Messages.create = _hooked_anthropic_create
    # async version similarly patches AsyncMessages.create
```

**Critical**: Pass the full `response` object to `tracker.track()`, **not** `response.usage`.
- `track()` internally calls `getattr(response, "usage", None)` to extract the usage object.
- Anthropic response has `response.usage.input_tokens` / `response.usage.output_tokens` (no `total_tokens`).
- `track()` handles this via: `elif hasattr(usage, "input_tokens"):`

## Anthropic Response Format

```python
# Anthropic SDK response (Message object)
response.usage.input_tokens   # int
response.usage.output_tokens  # int
# (no total_tokens field)
```

UsageTracker accumulates: `total = input_tokens + output_tokens`

## Verification

After installing, verify the hook is active:

```python
from anthropic.resources.messages.messages import Messages
assert Messages.create.__name__ == '_hooked_anthropic_create'
```

Or check the output `usage` field — if `token_source == "actual"`, the hook is working.

## Agents Affected

| Agent | Framework | Hook Required |
|-------|-----------|---------------|
| thinkdepthai | LangChain + ChatAnthropic | `install_anthropic_hooks()` ✅ |
| deerflow (deer-flow-v2) | LangChain + ChatAnthropic | `install_anthropic_hooks()` ✅ |
| aiq | LangChain + ChatAnthropic | `install_anthropic_hooks()` ✅ |
| auto_deep_research | litellm (anthropic/ prefix) | `install_litellm_hooks()` ✅ |
| deepresearchagent | Direct OpenAI SDK | `install_openai_hooks()` only |
| openrca | Direct OpenAI SDK | `install_openai_hooks()` only |
| mabc | Direct OpenAI SDK | `install_openai_hooks()` only |
| taskweaver | OpenAI SDK (streaming) | `install_openai_hooks()` with streaming fix |

## Background: Why ChatOpenAI Doesn't Work

Before this fix, a naive approach would be to switch `ChatAnthropic` → `ChatOpenAI` (which routes through OpenAI SDK and gets intercepted). **Do NOT do this.**

`ChatOpenAI` pointing to the shubiaobiao proxy's Anthropic models causes **Bedrock format 400 errors** — the proxy's Bedrock backend expects Anthropic message format but receives OpenAI format. This was the original bug fixed in `docs/plans/2026-03-13-agent-fix-progress.md`.

The correct fix is: keep `ChatAnthropic`, add `install_anthropic_hooks()`.

## Token Count Comparison (dataset_index=5)

| Agent | Estimated | Actual | Ratio |
|-------|-----------|--------|-------|
| thinkdepthai | ~36,842 | 462,263 | **12.5×** |
| deerflow | ~8,970 | 267,991 | **29.9×** |
| aiq | ~39,440 | 653,054 | **16.6×** |

The O(n²) growth in input tokens (full context re-sent each call) makes estimation fundamentally inaccurate for multi-turn agents.
