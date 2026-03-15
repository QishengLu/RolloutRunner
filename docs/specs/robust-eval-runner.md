# Spec: Robust Eval Runner

**目的**：修复快速失败写入 DB 的 bug，同时实现 AIMD 自适应并发和串行 agent 执行，保证断线后实验继续运行。

---

## 一、背景与问题

### 1.1 问题 A：快速失败污染 DB（最高优先级）

**现象**：某些 agent 在初始化阶段出错（API key 失效、依赖加载失败等），exit code 为 0，输出空 JSON `{"output": "", "trajectory": [], "error": "..."}` → `runner.py` 成功解析 → `db_writer.py` 将 `sample.stage = "rollout"` 写入 DB → `load_samples()` 只读 `stage='init'`，该样本**永远不会被重跑**。

**代码路径**（当前行为）：
```
runner.py:67  result = _parse_last_json(stdout) → 成功解析
runner.py:75  return AgentResult(output="", trajectory=[])  ← 没有验证
run_rollout.py:134  on_complete(result)
db_writer.py:33  sample.stage = "rollout"  ← BUG：垃圾数据也被写入
dataset.py:99   load_samples 只查 stage='init' → 该样本永久跳过
```

**正确行为**（exit ≠0 的路径，可以参考）：
```
runner.py:60  if proc.returncode != 0: return None
run_rollout.py:135  if result is None: counters["failure"] += 1; return
DB 未被修改，stage 保持 'init'，下次重跑自动补上
```

**字段说明**（避免混淆）：

| 层级 | 字段名 |
|------|--------|
| agent stdout JSON | `"output"`, `"trajectory"` （单数，所有 8 个 agent 均如此） |
| `AgentResult` dataclass | `.output`, `.trajectory` |
| DB 列名（db_writer.py 写入） | `sample.response`, `sample.trajectories` （复数） |

验证逻辑在 `runner.py` 里操作的是 **agent stdout JSON 字段**（`result.get("output")` / `result.get("trajectory")`），发生在写 DB 之前，和 DB 列名无关。

---

### 1.2 问题 B：SSH 断线导致实验中断

**现象**：直接在 shell 里跑 `python run_rollout.py`，SSH 断线后进程被 SIGHUP kill。

**方案**：用 tmux session 持久化，断线后 `tmux attach -t eval` 即可恢复。

---

### 1.3 问题 C：高并发打爆 API

**现象**：8 个 agent 同时跑，每个 `concurrency=15`，理论上同时有 120 个并发请求，shubiaobiao API 会 rate limit。

**方案**：
1. 串行执行：一个 agent 的所有样本跑完再跑下一个（消除 agent 间并发）
2. 单 agent 内使用 AIMD 自适应并发（慢启动从 1 开始，成功则渐增，失败则减半+退避）

---

## 二、变更说明

### 变更 1：`runner.py` — 修复空 output 不写 DB

**文件**：`RolloutRunner/src/runner.py`

**位置**：`run_agent()` 函数，紧接在 `_parse_last_json` 返回 None 的检查之后（当前第 67-73 行）。

**当前代码**（第 67-81 行）：
```python
result = _parse_last_json(stdout.decode().strip())
if result is None:
    logger.error(
        f"[sample {sample_id}] Failed to parse JSON from stdout: "
        f"{stdout.decode()[:300]}"
    )
    return None

return AgentResult(
    sample_id=sample_id,
    output=result.get("output", ""),
    trajectory=result.get("trajectory", []),
    time_cost=elapsed,
    usage=result.get("usage", {}),
)
```

**修改后**（在 `return None` 和 `return AgentResult(...)` 之间插入）：
```python
result = _parse_last_json(stdout.decode().strip())
if result is None:
    logger.error(
        f"[sample {sample_id}] Failed to parse JSON from stdout: "
        f"{stdout.decode()[:300]}"
    )
    return None

# 验证：exit 0 但 output 和 trajectory 均为空 → 视为失败，不写 DB
# 这防止快速失败的 agent（如初始化出错）把垃圾数据写入 DB 并锁死该样本
if not result.get("output") and not result.get("trajectory"):
    err_msg = result.get("error", "empty output and trajectory")
    logger.error(
        f"[sample {sample_id}] Agent exited 0 but produced no usable output: "
        f"{str(err_msg)[:300]}"
    )
    return None

return AgentResult(
    sample_id=sample_id,
    output=result.get("output", ""),
    trajectory=result.get("trajectory", []),
    time_cost=elapsed,
    usage=result.get("usage", {}),
)
```

---

### 变更 2：`runner.py` — AIMD 自适应并发

**文件**：`RolloutRunner/src/runner.py`

**位置**：在 `_parse_last_json()` 之后、`run_batch()` 之前，新增 `AdaptiveConcurrency` 类，并替换 `run_batch()` 的实现。

#### 2a. 新增类（插入在 `_parse_last_json` 函数之后）

```python
class AdaptiveConcurrency:
    """AIMD-style adaptive concurrency limiter.

    慢启动（Slow Start）：从 capacity=1 开始，每连续 10 次成功后 +1。
    乘法减少（Multiplicative Decrease）：任何失败后 capacity = max(1, capacity // 2)。
    退避（Backoff）：失败后 sleep backoff 秒（持续占用该 slot），阻止后续请求，
                   初始 5s，每次失败翻倍，成功后缓慢减半，上限 120s。
    上限：max_capacity（来自 YAML 的 concurrency 字段）。
    """

    def __init__(self, max_capacity: int) -> None:
        self.max_capacity = max(1, max_capacity)
        self.capacity = 1
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
                    self._success_streak >= 10
                    and self.capacity < self.max_capacity
                ):
                    self.capacity += 1
                    self._success_streak = 0
                    self._backoff = max(5.0, self._backoff * 0.5)
                    logger.info(
                        f"[AIMD] Concurrency ↑ {self.capacity} "
                        f"(backoff={self._backoff:.0f}s)"
                    )
            else:
                new_cap = max(1, self.capacity // 2)
                if new_cap < self.capacity:
                    logger.warning(
                        f"[AIMD] Concurrency ↓ {self.capacity} → {new_cap} "
                        f"(backoff={self._backoff:.0f}s)"
                    )
                self.capacity = new_cap
                self._success_streak = 0
            self._cond.notify_all()

    async def backoff_on_failure(self, success: bool) -> None:
        """失败时 sleep（期间继续占用 slot，阻止新请求涌入）。"""
        if not success:
            await asyncio.sleep(self._backoff)
            self._backoff = min(120.0, self._backoff * 2)
```

#### 2b. 替换 `run_batch()` 实现

**当前 `run_batch()` 实现**（第 104-129 行）：
```python
async def run_batch(
    samples: list[dict],
    cmd: list[str],
    cwd: str,
    timeout: float,
    concurrency: int,
    env: dict[str, str] | None = None,
    on_complete: "Callable[[AgentResult | None], None] | None" = None,
) -> list[AgentResult | None]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: dict) -> AgentResult | None:
        async with semaphore:
            result = await run_agent(
                sample_id=item["id"],
                payload=item["payload"],
                cmd=cmd,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
            if on_complete is not None:
                on_complete(result)
            return result

    return await asyncio.gather(*[_run(s) for s in samples])
```

**替换为**：
```python
async def run_batch(
    samples: list[dict],
    cmd: list[str],
    cwd: str,
    timeout: float,
    concurrency: int,
    env: dict[str, str] | None = None,
    on_complete: "Callable[[AgentResult | None], None] | None" = None,
) -> list[AgentResult | None]:
    ac = AdaptiveConcurrency(max_capacity=concurrency)
    logger.info(f"[AIMD] Starting with capacity=1, max={concurrency}")

    async def _run(item: dict) -> AgentResult | None:
        await ac.acquire()
        result = None
        try:
            result = await run_agent(
                sample_id=item["id"],
                payload=item["payload"],
                cmd=cmd,
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
            success = result is not None
            await ac.release(success)

    return await asyncio.gather(*[_run(s) for s in samples])
```

**注意**：`finally` 块里的 `success = result is not None` 在 `result=None`（初始值）的异常情况下也能正确标记失败。`backoff_on_failure` 和 `on_complete` 在 `finally` 之前执行，确保 slot 在 DB 写入完成后才释放。

---

### 变更 3：YAML 并发上限下调

**文件**：以下 8 个文件（均为 `RolloutRunner/configs/agents/*.yaml`）

```
thinkdepthai.yaml
deerflow.yaml
auto_deep_research.yaml
deepresearchagent.yaml
aiq.yaml
taskweaver.yaml
openrca.yaml
mabc.yaml
```

**修改**：每个文件中 `concurrency: 15` → `concurrency: 5`

这是 AIMD 的 `max_capacity` 上限。慢启动从 1 开始，平稳后最多维持 5 个并发请求。

---

### 变更 4：新建两个运行脚本

#### 4a. 串行主脚本：`RolloutRunner/run_eval_sequential.sh`

一个 agent 的所有样本跑完（`run_rollout.py` 退出）再启动下一个。

```bash
#!/bin/bash
# run_eval_sequential.sh — 依次跑 8 个 agent 的 rollout
# 用法：bash run_eval_sequential.sh
# 断线后：tmux attach -t eval → 在 runner window 里确认仍在跑

set -euo pipefail

export UTU_DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"

ROLLOUT_DIR="/home/nn/SOTA-agents/RolloutRunner"
AGENTS=(thinkdepthai deerflow auto_deep_research deepresearchagent aiq taskweaver openrca mabc)

mkdir -p "$ROLLOUT_DIR/logs"

for NAME in "${AGENTS[@]}"; do
    EXP="${NAME}-claude-sonnet-4.6"
    echo ""
    echo "========================================"
    echo "=== AGENT: $NAME"
    echo "=== START: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    cd "$ROLLOUT_DIR"
    uv run python scripts/run_rollout.py \
        --agent "$NAME" \
        --source_exp_id "$EXP" \
        2>&1 | tee "logs/${NAME}-4.6.log"

    echo "=== DONE: $NAME at $(date '+%Y-%m-%d %H:%M:%S') ==="
done

echo ""
echo "========================================"
echo "=== ALL 8 AGENTS COMPLETE ==="
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "========================================"
```

#### 4b. tmux 启动器：`RolloutRunner/launch_eval.sh`

```bash
#!/bin/bash
# launch_eval.sh — 在 tmux 里启动串行实验 + 监控 window
# 用法：bash launch_eval.sh
# 恢复：tmux attach -t eval

SESSION="eval"
ROLLOUT_DIR="/home/nn/SOTA-agents/RolloutRunner"
DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "runner"

# runner window：串行跑所有 agent
tmux send-keys -t "$SESSION:runner" \
    "bash $ROLLOUT_DIR/run_eval_sequential.sh" Enter

# monitor window：每 30s 刷新 DB 进度
tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" \
    "watch -n 30 \"psql '$DB_URL' -c \\\"SELECT exp_id, stage, COUNT(*) as cnt FROM evaluation_data GROUP BY exp_id, stage ORDER BY exp_id, stage\\\"\"" Enter

tmux select-window -t "$SESSION:runner"
tmux attach -t "$SESSION"
```

---

## 三、前置步骤（实验前一次性执行）

在运行 `launch_eval.sh` 之前，需要确保每个 agent 在 DB 里有 `stage='init'` 的样本：

```bash
cd /home/nn/SOTA-agents/RCAgentEval
export UTU_DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"

for agent in thinkdepthai deerflow auto_deep_research deepresearchagent aiq taskweaver openrca mabc; do
    echo "Preprocessing: $agent"
    uv run python scripts/preprocess_only.py --exp_id "${agent}-claude-sonnet-4.6"
done
```

验证：
```sql
SELECT exp_id, stage, COUNT(*)
FROM evaluation_data
WHERE stage='init'
GROUP BY exp_id, stage
ORDER BY exp_id;
-- 每个 exp_id 应有 500 行
```

---

## 四、日常操作手册

| 操作 | 命令 |
|------|------|
| 启动实验 | `bash /home/nn/SOTA-agents/RolloutRunner/launch_eval.sh` |
| 断线后恢复监控 | `tmux attach -t eval` |
| 切换到 runner window | tmux 内按 `Ctrl+b` 然后按 `0` |
| 切换到 monitor window | tmux 内按 `Ctrl+b` 然后按 `1` |
| 临时调大某 agent 并发上限 | 编辑 `configs/agents/<name>.yaml`：`concurrency: 8`（对当前 agent 的下一轮生效） |
| 某 agent 中途失败，手动补跑 | `tmux attach -t eval` → runner window → `↑ Enter`（重跑同一条命令，自动跳过已完成样本） |
| 查看某 agent 的历史日志 | `cat RolloutRunner/logs/thinkdepthai-4.6.log` |
| 查看实时 DB 进度 | monitor window，或 `psql ... -c "SELECT ..."` |

---

## 五、验证用例

### 5.1 空 output 不写 DB

```bash
# 构造一个快速失败的 mock agent，exit 0 但输出空 JSON
echo '{"output":"","trajectory":[],"error":"mock init error"}' > /tmp/mock_out.txt

# 在 DB 里找一个 stage='init' 的样本，记下其 id，然后模拟调用
# 验证：运行后该样本 stage 仍为 'init'
psql "$UTU_DB_URL" -c "SELECT id, stage FROM evaluation_data WHERE id=<sample_id>"
# 期望：stage='init'（未被修改）
```

### 5.2 AIMD 行为验证

跑任意一个 agent，在日志里观察：
```
[AIMD] Starting with capacity=1, max=5
# 前 10 次成功后：
[AIMD] Concurrency ↑ 2 (backoff=5s)
# 再 10 次成功后：
[AIMD] Concurrency ↑ 3 (backoff=5s)
# 某次失败（API 限流/超时）：
[AIMD] Concurrency ↓ 3 → 1 (backoff=5s)
```

### 5.3 串行顺序验证

在 runner window 里确认输出顺序（`thinkdepthai` 的 `DONE` 行出现后，才出现 `deerflow` 的 `START` 行）：
```
=== DONE: thinkdepthai at 2026-03-15 10:30:00 ===

========================================
=== AGENT: deerflow
=== START: 2026-03-15 10:30:01
========================================
```

### 5.4 断线保护验证

```bash
# 启动实验
bash launch_eval.sh
# 断开 SSH（关闭终端窗口）
# 重新 SSH 进来
tmux attach -t eval
# 确认 runner window 仍在输出进度日志
```

---

## 六、文件变更汇总

| 文件 | 变更类型 | 内容 |
|------|---------|------|
| `RolloutRunner/src/runner.py` | 修改 | 1. 空 output+trajectory 验证（return None）；2. 新增 `AdaptiveConcurrency` 类；3. 替换 `run_batch()` 使用 AIMD |
| `RolloutRunner/configs/agents/thinkdepthai.yaml` | 修改 | `concurrency: 15` → `concurrency: 5` |
| `RolloutRunner/configs/agents/deerflow.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/auto_deep_research.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/deepresearchagent.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/aiq.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/taskweaver.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/openrca.yaml` | 修改 | 同上 |
| `RolloutRunner/configs/agents/mabc.yaml` | 修改 | 同上 |
| `RolloutRunner/run_eval_sequential.sh` | 新建 | 串行跑 8 个 agent 的 bash 脚本 |
| `RolloutRunner/launch_eval.sh` | 新建 | tmux 启动器（runner + monitor window） |
