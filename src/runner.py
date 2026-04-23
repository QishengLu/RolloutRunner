"""asyncio subprocess runner，通过 stdin/stdout 与 agent 通信。"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    sample_id: int
    output: str
    trajectory: list = field(default_factory=list)
    time_cost: float = 0.0
    usage: dict = field(default_factory=dict)  # token usage from agent
    extra_meta: dict = field(default_factory=dict)  # optional per-agent meta (e.g. taskweaver plugin_calls_raw)


async def run_agent(
    *,
    sample_id: int,
    payload: dict[str, Any],
    cmd: list[str],
    cwd: str,
    timeout: float,
    env: dict[str, str] | None = None,
) -> AgentResult | None:
    stdin_data = json.dumps(payload, ensure_ascii=False).encode()
    start = time.time()
    logger.info(f"[sample {sample_id}] Starting agent subprocess...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(f"[sample {sample_id}] Timeout after {timeout}s")
            return None

        elapsed = time.time() - start
        logger.info(f"[sample {sample_id}] Agent finished in {elapsed:.1f}s, exit_code={proc.returncode}")

        if stderr:
            logger.debug(f"[sample {sample_id}] stderr: {stderr.decode()[:200]}")

        if proc.returncode != 0:
            logger.error(
                f"[sample {sample_id}] Exit code {proc.returncode}. "
                f"stderr: {stderr.decode()[:1000]}"
            )
            return None

        result = _parse_last_json(stdout.decode().strip())
        if result is None:
            logger.error(
                f"[sample {sample_id}] Failed to parse JSON from stdout: "
                f"{stdout.decode()[:300]}"
            )
            return None

        # 验证：exit 0 但 output 和 trajectory 均为空 → 视为失败，不写 DB
        # 防止快速失败的 agent（初始化出错等）把垃圾数据写入 DB 并锁死该样本
        if not result.get("output") and not result.get("trajectory"):
            err_msg = result.get("error", "empty output and trajectory")
            logger.error(
                f"[sample {sample_id}] Agent exited 0 but produced no usable output: "
                f"{str(err_msg)[:300]}"
            )
            return None

        # Collect any extra fields the agent emitted beyond the standard
        # {output, trajectory, usage}. Persisted verbatim to DB meta so
        # per-agent forensic data (e.g. TaskWeaver plugin_calls_raw /
        # taskweaver_session_id / shadow_fallback) survives the pipeline.
        _STD_KEYS = {"output", "trajectory", "usage", "error"}
        extra_meta = {k: v for k, v in result.items() if k not in _STD_KEYS}

        return AgentResult(
            sample_id=sample_id,
            output=result.get("output", ""),
            trajectory=result.get("trajectory", []),
            time_cost=elapsed,
            usage=result.get("usage", {}),
            extra_meta=extra_meta,
        )

    except Exception as e:
        logger.error(f"[sample {sample_id}] Unexpected error: {e}", exc_info=True)
        return None


def _parse_last_json(text: str) -> dict | None:
    """从文本中找最后一行有效 JSON（跳过调试输出行）。"""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class AdaptiveConcurrency:
    """AIMD-style adaptive concurrency limiter.

    慢启动（Slow Start）：从 capacity=1 开始，每连续 3 次成功后 +1。
    温和减少（Subtractive Decrease）：任何失败后 capacity = max(1, capacity - 1)。
    退避（Backoff）：失败后 sleep backoff 秒（持续占用该 slot，阻止新请求涌入），
                   固定 5s，不指数增长（RCA 场景的失败多是 timeout 而非 rate limit）。
    上限：max_capacity（来自 YAML 的 concurrency 字段）。

    设计理由：
    - RCA agent 每个 sample 耗时 60-1000s，10 次成功爬一级需要 1-3 小时太慢
    - 失败多是 timeout/agent 异常，非 API rate limit，不需要激进减半
    - 3 次成功 +1 约 3-15 分钟到 max_capacity，更合理
    """

    def __init__(self, max_capacity: int, initial_capacity: int = 3) -> None:
        self.max_capacity = max(1, max_capacity)
        self.capacity = min(initial_capacity, self.max_capacity)
        self._active = 0
        self._success_streak = 0
        self._backoff = 5.0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        """等待可用 slot。"""
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self.capacity)
            self._active += 1

    async def release(self, success: bool) -> None:
        """释放 slot 并根据结果调整 capacity。"""
        async with self._cond:
            self._active -= 1
            if success:
                self._success_streak += 1
                if (
                    self._success_streak >= 3
                    and self.capacity < self.max_capacity
                ):
                    self.capacity += 1
                    self._success_streak = 0
                    logger.info(
                        f"[AIMD] Concurrency ↑ {self.capacity} "
                        f"(streak reset, max={self.max_capacity})"
                    )
            else:
                new_cap = max(1, self.capacity - 1)
                if new_cap < self.capacity:
                    logger.warning(
                        f"[AIMD] Concurrency ↓ {self.capacity} → {new_cap} "
                        f"(failure, backoff={self._backoff:.0f}s)"
                    )
                self.capacity = new_cap
                self._success_streak = 0
            self._cond.notify_all()

    async def backoff_on_failure(self, success: bool) -> None:
        """失败时 sleep（期间继续占用 slot，限速）。"""
        if not success:
            await asyncio.sleep(self._backoff)


async def run_batch(
    samples: list[dict],
    cmd: list[str],
    cwd: str,
    timeout: float,
    concurrency: int,
    initial_concurrency: int | None = None,
    env: dict[str, str] | None = None,
    on_complete: "Callable[[AgentResult | None], None] | None" = None,
) -> list[AgentResult | None]:
    import os as _os
    _env_initial = _os.environ.get("AIMD_INITIAL_CAPACITY")
    _init_cap = int(_env_initial) if _env_initial else (initial_concurrency or 3)
    ac = AdaptiveConcurrency(
        max_capacity=concurrency,
        initial_capacity=_init_cap,
    )
    logger.info(f"[AIMD] Starting with capacity={ac.capacity}, max={concurrency}")

    async def _run(item: dict) -> AgentResult | None:
        await ac.acquire()
        result = None
        try:
            result = await run_agent(
                sample_id=item["id"],
                payload=item["payload"],
                cmd=item.get("cmd", cmd),
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
            success = result is not None
            # 失败时 sleep（持续占用 slot，限速），再通知 on_complete
            await ac.backoff_on_failure(success)
            if on_complete is not None:
                on_complete(result)
            return result
        finally:
            # finally 里 result 可能仍为 None（异常情况），正确标记失败
            await ac.release(result is not None)

    return await asyncio.gather(*[_run(s) for s in samples])
