# RolloutRunner + ThinkDepthAI agent_runner 实施计划


> ⚠️ **此文档为历史实施记录。实际实现与计划有以下差异（以 EVAL_RUNBOOK.md 为准）：**
> - stdin 新增第 6 个字段 `data_dir`（从 augmented_question 正则提取，非配置文件）
> - `dataset.py` 使用本地 EvaluationSample 副本，不 sys.path.insert RCAgentEval
> - `agent_runner.py` 新增 `strip_markdown_json()` 处理 LLM 输出的 markdown 包裹
> - Judge 使用 `rejudge_samples.py`，不用 `run_eval_judge.py`
> - DB 路径统一用 4 斜杠绝对路径

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建 RolloutRunner 项目，为 Deep_Research（thinkdepthai）接入 RCAgentEval 测评流水线。

**Architecture:** RolloutRunner 通过 asyncio subprocess 调用 `agent_runner.py`，stdin 传入 5 字段 JSON（question + 4 个格式化好的 prompt），stdout 接收 CausalGraph JSON + OpenAI 格式 trajectory。`agent_runner.py` 用参数化闭包重建与原始相同拓扑的 LangGraph 图，注入 RCA prompts 和工具集，结果写回 RCAgentEval DB。

**Tech Stack:** Python 3.11+, uv, sqlmodel, pyyaml, asyncio（RolloutRunner）；langgraph, langchain-openai, dotenv（agent_runner）

---

## 关键路径

| 项目 | 路径 |
|------|------|
| RolloutRunner | `/home/nn/SOTA-agents/RolloutRunner` |
| Deep_Research | `/home/nn/SOTA-agents/Deep_Research` |
| RCAgentEval | `/home/nn/SOTA-agents/RCAgentEval` |
| RCA prompts（本地副本） | `RolloutRunner/configs/prompts/rca.yaml` |
| EvaluationSample 模型 | `RCAgentEval/utu/db/eval_datapoint.py` |
| DB 连接 | 环境变量 `UTU_DB_URL`（与 RCAgentEval 共用） |

**stdin/stdout 接口：**
```
stdin:  { "question": str, "system_prompt": str, "user_prompt": str,
          "compress_system_prompt": str, "compress_user_prompt": str }
stdout: { "output": str (CausalGraph JSON), "trajectory": list }
```

---

## Task 1: 初始化 RolloutRunner 项目骨架

**Files:**
- Create: `RolloutRunner/pyproject.toml`
- Create: `RolloutRunner/configs/agents/thinkdepthai.yaml`
- Create: `RolloutRunner/src/__init__.py`
- Create: `RolloutRunner/scripts/__init__.py`
- Already exists: `RolloutRunner/configs/prompts/rca.yaml`（已从 RCAgentEval 复制）

**Step 1: 创建目录和空文件**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
mkdir -p src scripts
touch src/__init__.py scripts/__init__.py
```

**Step 2: 创建 `pyproject.toml`**

```toml
[project]
name = "rollout-runner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sqlmodel>=0.0.21",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "sqlalchemy>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

**Step 3: 创建 `configs/agents/thinkdepthai.yaml`**

```yaml
name: thinkdepthai
cmd: ["uv", "run", "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/Deep_Research
exp_id: rollout_thinkdepthai
model_name: openai:gpt-5
agent_type: thinkdepthai
concurrency: 2
timeout: 600
```

**Step 4: 安装依赖**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
uv sync
```

Expected: `.venv` 目录生成，无报错。

**Step 5: Commit**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
git init
git add .
git commit -m "chore: init RolloutRunner project skeleton with prompts"
```

---

## Task 2: 实现 `src/dataset.py`

**Files:**
- Create: `RolloutRunner/src/dataset.py`

**背景：**
- `EvaluationSample` 在 `RCAgentEval/utu/db/eval_datapoint.py`，表名 `evaluation_data`
- 字段：`id`, `augmented_question`, `correct_answer`, `source`, `exp_id`, `stage`
- 通过 `sys.path.insert` 引用 RCAgentEval 的模型定义，避免重复声明

**Step 1: 写 `src/dataset.py`**

```python
"""从 RCAgentEval DB 读取待测评样本。"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, "/home/nn/SOTA-agents/RCAgentEval")

from sqlmodel import Session, create_engine, select
from utu.db.eval_datapoint import EvaluationSample


@dataclass
class SampleRecord:
    id: int
    augmented_question: str
    correct_answer: str
    source: str


def get_engine():
    db_url = os.environ.get("UTU_DB_URL")
    if not db_url:
        raise ValueError("UTU_DB_URL 环境变量未设置")
    return create_engine(db_url)


def load_samples(source_exp_id: str) -> list[SampleRecord]:
    """加载指定 exp_id 下 stage='init' 的样本。"""
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(EvaluationSample).where(
            EvaluationSample.exp_id == source_exp_id,
            EvaluationSample.stage == "init",
        )
        rows = session.exec(stmt).all()

    return [
        SampleRecord(
            id=r.id,
            augmented_question=r.augmented_question or "",
            correct_answer=r.correct_answer or "",
            source=r.source,
        )
        for r in rows
    ]
```

**Step 2: 冒烟验证**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
uv run python -c "
from src.dataset import load_samples
samples = load_samples('rcabench_evaluation')
print(f'Found {len(samples)} samples')
if samples:
    print('q[:100]:', samples[0].augmented_question[:100])
"
```

Expected: 打印样本数和问题前缀（如 DB 中有数据）。

**Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: dataset loader from RCAgentEval DB"
```

---

## Task 3: 实现 `src/runner.py`

**Files:**
- Create: `RolloutRunner/src/runner.py`

**关键点：**
- subprocess 通过 stdin 传 JSON payload，从 stdout 读结果
- stdout 可能混有调试输出，`_parse_last_json` 从后往前找最后一行有效 JSON
- 超时 → kill subprocess → 返回 None
- 非零退出码或 JSON 解析失败 → 记录 stderr → 返回 None

**Step 1: 写 `src/runner.py`**

```python
"""asyncio subprocess runner，通过 stdin/stdout 与 agent 通信。"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    sample_id: int
    output: str
    trajectory: list = field(default_factory=list)
    time_cost: float = 0.0


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

        if proc.returncode != 0:
            logger.error(
                f"[sample {sample_id}] Exit code {proc.returncode}. "
                f"stderr: {stderr.decode()[:500]}"
            )
            return None

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


async def run_batch(
    samples: list[dict],
    cmd: list[str],
    cwd: str,
    timeout: float,
    concurrency: int,
    env: dict[str, str] | None = None,
) -> list[AgentResult | None]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: dict) -> AgentResult | None:
        async with semaphore:
            return await run_agent(
                sample_id=item["id"],
                payload=item["payload"],
                cmd=cmd,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )

    return await asyncio.gather(*[_run(s) for s in samples])
```

**Step 2: Commit**

```bash
git add src/runner.py
git commit -m "feat: asyncio subprocess runner with semaphore"
```

---

## Task 4: 实现 `src/db_writer.py`

**Files:**
- Create: `RolloutRunner/src/db_writer.py`

**Step 1: 写 `src/db_writer.py`**

```python
"""将 agent 结果写回 RCAgentEval EvaluationSample。"""
import datetime
import json
import logging
import sys

sys.path.insert(0, "/home/nn/SOTA-agents/RCAgentEval")

from sqlmodel import Session
from utu.db.eval_datapoint import EvaluationSample

from src.dataset import get_engine
from src.runner import AgentResult

logger = logging.getLogger(__name__)


def write_result(
    *,
    result: AgentResult,
    exp_id: str,
    agent_type: str,
    model_name: str,
) -> bool:
    engine = get_engine()
    with Session(engine) as session:
        sample = session.get(EvaluationSample, result.sample_id)
        if sample is None:
            logger.error(f"Sample {result.sample_id} not found in DB")
            return False

        sample.response = result.output
        sample.trajectories = json.dumps(result.trajectory, ensure_ascii=False)
        sample.time_cost = result.time_cost
        sample.stage = "rollout"
        sample.exp_id = exp_id
        sample.agent_type = agent_type
        sample.model_name = model_name
        sample.updated_at = datetime.datetime.now()

        session.add(sample)
        session.commit()

    logger.info(f"[sample {result.sample_id}] Written to DB (stage=rollout)")
    return True


def write_batch(
    results: list[AgentResult | None],
    exp_id: str,
    agent_type: str,
    model_name: str,
) -> tuple[int, int]:
    success, failure = 0, 0
    for result in results:
        if result is None:
            failure += 1
            continue
        ok = write_result(
            result=result, exp_id=exp_id, agent_type=agent_type, model_name=model_name
        )
        success += 1 if ok else 0
        failure += 0 if ok else 1
    return success, failure
```

**Step 2: Commit**

```bash
git add src/db_writer.py
git commit -m "feat: DB writer for rollout results"
```

---

## Task 5: 实现 `scripts/run_rollout.py`

**Files:**
- Create: `RolloutRunner/scripts/run_rollout.py`

**关键点：**
- prompts 从 `RolloutRunner/configs/prompts/rca.yaml` 读取（本地副本）
- `RCA_ANALYSIS_SP` 含 `{date}` 占位符，format 时传入今天日期
- `RCA_ANALYSIS_UP` / `COMPRESS_FINDINGS_UP` 含 `{incident_description}`，传入 `augmented_question`
- `COMPRESS_FINDINGS_SP` 含 `{date}`
- `source_exp_id` 参数：从哪个已有 exp_id 的样本集读取（通常是 `rcabench_evaluation`）

**Step 1: 写 `scripts/run_rollout.py`**

```python
#!/usr/bin/env python
"""RolloutRunner 入口脚本。"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_samples
from src.db_writer import write_batch
from src.runner import run_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "configs" / "prompts" / "rca.yaml"
AGENTS_DIR = Path(__file__).parent.parent / "configs" / "agents"


def load_agent_config(agent_name: str) -> dict:
    path = AGENTS_DIR / f"{agent_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_rca_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_payload(question: str, prompts: dict) -> dict:
    today = date.today().isoformat()
    return {
        "question": question,
        "system_prompt": prompts["RCA_ANALYSIS_SP"].format(date=today),
        "user_prompt": prompts["RCA_ANALYSIS_UP"].format(incident_description=question),
        "compress_system_prompt": prompts["COMPRESS_FINDINGS_SP"].format(date=today),
        "compress_user_prompt": prompts["COMPRESS_FINDINGS_UP"].format(
            incident_description=question
        ),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="configs/agents/<name>.yaml")
    parser.add_argument(
        "--source_exp_id",
        default="rcabench_evaluation",
        help="从哪个 exp_id 的 stage=init 样本读取",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制样本数（调试用）")
    args = parser.parse_args()

    cfg = load_agent_config(args.agent)
    prompts = load_rca_prompts()

    logger.info(f"Agent: {cfg['name']}  exp_id: {cfg['exp_id']}")
    logger.info(f"Reading from exp_id={args.source_exp_id}, stage=init")

    samples = load_samples(args.source_exp_id)
    if args.limit:
        samples = samples[: args.limit]
    logger.info(f"Loaded {len(samples)} samples")

    if not samples:
        logger.warning("No samples found, exiting.")
        return

    tasks = [
        {"id": s.id, "payload": build_payload(s.augmented_question, prompts)}
        for s in samples
    ]

    results = await run_batch(
        samples=tasks,
        cmd=cfg["cmd"],
        cwd=cfg["cwd"],
        timeout=cfg["timeout"],
        concurrency=cfg["concurrency"],
        env=dict(os.environ),
    )

    success, failure = write_batch(
        results=results,
        exp_id=cfg["exp_id"],
        agent_type=cfg["agent_type"],
        model_name=cfg["model_name"],
    )
    logger.info(f"Done. success={success}, failure={failure}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add scripts/run_rollout.py
git commit -m "feat: run_rollout.py entry script"
```

---

## Task 6: 实现 `Deep_Research/agent_runner.py`

**Files:**
- Create: `Deep_Research/agent_runner.py`

**Deep_Research 内部结构说明：**

| 模块 | 路径 | 用途 |
|------|------|------|
| `research_agent_prompt` | `deep_research/prompts.py` | 原始研究 system prompt（叠加保留） |
| `ResearcherState` | `deep_research/state_research.py` | 图状态定义 |
| `ResearcherOutputState` | `deep_research/state_research.py` | 图输出状态 |
| `think_tool` | `deep_research/utils.py` | 反思工具 |
| parquet 工具 | `src/rca_tools.py` | `list_tables_in_directory`, `get_schema`, `query_parquet_files` |
| `.env` | `Deep_Research/.env` | OPENAI_API_KEY 等 |

**LangChain → OpenAI 消息格式转换：**

```
HumanMessage  → {"role": "user", "content": str}
AIMessage     → {"role": "assistant", "content": str,
                 "tool_calls": [{"id": tc["id"], "type": "function",
                   "function": {"name": tc["name"],
                                "arguments": json.dumps(tc["args"])}}]}
ToolMessage   → {"role": "tool", "content": str, "tool_call_id": str}
其他类型       → 跳过（不输出）
```

**Step 1: 写 `Deep_Research/agent_runner.py`**

```python
#!/usr/bin/env python
"""
agent_runner.py — ThinkDepthAI RCA 测评接口

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式) }
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal

from deep_research.prompts import research_agent_prompt
from deep_research.state_research import ResearcherOutputState, ResearcherState
from deep_research.utils import think_tool
from src.rca_tools import get_schema, list_tables_in_directory, query_parquet_files

# ── 工具集（与 RCA_ANALYSIS_SP 描述一致，去掉 tavily_search）────────────────
RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
RCA_TOOLS_BY_NAME = {t.name: t for t in RCA_TOOLS}


# ── 节点工厂（闭包注入 prompt/tools，保持原拓扑）─────────────────────────────

def make_llm_call(combined_system_prompt: str):
    model = init_chat_model(model="openai:gpt-5")
    model_with_tools = model.bind_tools(RCA_TOOLS)

    def llm_call(state: ResearcherState):
        return {
            "researcher_messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=combined_system_prompt)]
                    + state["researcher_messages"]
                )
            ]
        }

    return llm_call


def tool_node(state: ResearcherState):
    tool_calls = state["researcher_messages"][-1].tool_calls
    outputs = []
    for tc in tool_calls:
        tool = RCA_TOOLS_BY_NAME[tc["name"]]
        result = tool.invoke(tc["args"])
        outputs.append(ToolMessage(content=result, name=tc["name"], tool_call_id=tc["id"]))
    return {"researcher_messages": outputs}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    return "tool_node" if state["researcher_messages"][-1].tool_calls else "compress_research"


def make_compress_research(compress_sp: str, compress_up: str):
    compress_model = init_chat_model(model="openai:gpt-5", max_tokens=32000)

    def compress_research(state: ResearcherState) -> dict:
        messages = (
            [SystemMessage(content=compress_sp)]
            + state.get("researcher_messages", [])
            + [HumanMessage(content=compress_up)]
        )
        response = compress_model.invoke(messages)
        raw_notes = [
            str(m.content)
            for m in filter_messages(
                state["researcher_messages"], include_types=["tool", "ai"]
            )
        ]
        return {
            "compressed_research": str(response.content),
            "raw_notes": ["\n".join(raw_notes)],
        }

    return compress_research


def build_agent(combined_sp: str, compress_sp: str, compress_up: str):
    """与原始 research_agent.py 完全相同的图拓扑，仅 prompt/tools 不同。"""
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("llm_call", make_llm_call(combined_sp))
    builder.add_node("tool_node", tool_node)
    builder.add_node("compress_research", make_compress_research(compress_sp, compress_up))
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", "compress_research": "compress_research"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("compress_research", END)
    return builder.compile()


# ── LangChain → OpenAI 格式转换 ──────────────────────────────────────────────

def to_openai_message(msg) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}

    if isinstance(msg, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        entry: dict = {"role": "assistant", "content": str(msg.content) if msg.content else ""}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    if isinstance(msg, ToolMessage):
        return {"role": "tool", "content": str(msg.content), "tool_call_id": msg.tool_call_id}

    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [m for msg in messages if (m := to_openai_message(msg)) is not None]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    payload = json.loads(sys.stdin.read())

    question = payload["question"]
    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    compress_sp = payload["compress_system_prompt"]
    compress_up = payload["compress_user_prompt"]

    # 叠加：保留 Deep_Research 原有推理风格，追加 RCA 专项指令
    combined_sp = research_agent_prompt + "\n\n---\n\n" + system_prompt

    agent = build_agent(combined_sp, compress_sp, compress_up)

    initial_state = {"researcher_messages": [HumanMessage(content=user_prompt)]}

    all_messages: list = []
    compressed_research = ""

    for event in agent.stream(initial_state, config={"recursion_limit": 100}):
        for key, value in event.items():
            if not isinstance(value, dict):
                continue
            if "researcher_messages" in value:
                all_messages.extend(value["researcher_messages"])
            if "compressed_research" in value:
                compressed_research = value["compressed_research"]

    result = {
        "output": compressed_research,
        "trajectory": convert_trajectory(all_messages),
    }
    # 单行输出，runner._parse_last_json 从末行解析
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

**Step 2: 冒烟测试（需要 OPENAI_API_KEY）**

```bash
cd /home/nn/SOTA-agents/Deep_Research
echo '{"question":"test q","system_prompt":"You are RCA.","user_prompt":"Analyze: test q","compress_system_prompt":"Summarize.","compress_user_prompt":"Output JSON now."}' \
  | uv run python agent_runner.py
```

Expected: stdout 最后一行为 `{"output": "...", "trajectory": [...]}` 格式的 JSON。

---

## Task 7: 端到端单条样本验证

**前提：** `UTU_DB_URL` 和 `OPENAI_API_KEY` 已配置，DB 中有 `stage=init` 样本。

**Step 1: 单条 dry-run**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
uv run python scripts/run_rollout.py \
  --agent thinkdepthai \
  --source_exp_id rcabench_evaluation \
  --limit 1
```

Expected:
```
INFO: Loaded 1 samples
INFO: [sample <id>] Written to DB (stage=rollout)
INFO: Done. success=1, failure=0
```

**Step 2: 验证 DB 写入**

```bash
cd /home/nn/SOTA-agents/RCAgentEval
uv run python -c "
import os
from sqlmodel import Session, create_engine, select
from utu.db.eval_datapoint import EvaluationSample
engine = create_engine(os.environ['UTU_DB_URL'])
with Session(engine) as s:
    rows = s.exec(select(EvaluationSample).where(
        EvaluationSample.exp_id == 'rollout_thinkdepthai',
        EvaluationSample.stage == 'rollout',
    )).all()
    for r in rows:
        print('stage:', r.stage, '| response[:80]:', (r.response or '')[:80])
        print('trajectory sample:', (r.trajectories or '')[:100])
"
```

Expected: `response` 以 `{` 开头（CausalGraph JSON），`trajectories` 含 `role` 字段。

**Step 3: 验证 judge 可解析**

```bash
cd /home/nn/SOTA-agents/RCAgentEval
uv run python scripts/rejudge_samples.py
```

Expected: 打印 `✓ Correct: True/False, Score: ...`，无 `JSON decode error`。

**Step 4: 全量跑通**

```bash
cd /home/nn/SOTA-agents/RolloutRunner
uv run python scripts/run_rollout.py --agent thinkdepthai --source_exp_id rcabench_evaluation
```

---

## 常见问题

| 现象 | 排查 |
|------|------|
| `output` 为空 | agent_runner stream 循环未正确收集 `compressed_research`，加 print 调试 event |
| `JSON decode error` in judge | compress 输出含 markdown fence（` ```json ` ），检查 COMPRESS_FINDINGS_UP 是否要求纯 JSON |
| `failure=N` in runner | 查看 stderr 输出（runner.py 已记录） |
| `tool not found` | RCA_TOOLS_BY_NAME 未包含该 tool，检查 tool.name 与 AIMessage.tool_calls 里的 name 是否一致 |
| `No samples found` | `source_exp_id` 不匹配或样本未完成 preprocess（stage 不是 init） |
