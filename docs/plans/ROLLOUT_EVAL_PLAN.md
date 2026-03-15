# RolloutRunner 测评项目计划

## 背景

在 `/home/nn/SOTA-agents/RolloutRunner/` 下创建独立的 `RolloutRunner` 项目，专门为每个 agent 生成 rollout 数据，与各 agent 项目和 RCAgentEval 完全解耦。

### 设计原则

- **解耦**：RolloutRunner 通过 subprocess 调用各 agent，不 import 任何 agent 代码
- **统一输入**：所有 agent 接收相同的 `augmented_question`（来自 RCAgentEval 预处理）
- **统一 prompt**：RCA prompts 本地副本存于 `configs/prompts/rca.yaml`，格式化后通过 stdin 传给各 agent，各 agent 无需依赖 RCAgentEval 路径
- **统一输出**：所有 agent 输出 CausalGraph JSON，写入 RCAgentEval 的 DB，直接复用其 judge pipeline
- **环境隔离**：每个 agent 用自己的环境（uv/conda/venv），互不干扰

### 数据流

```
RCAgentEval DB (stage="init", augmented_question 已生成)
        ↓
RolloutRunner 读取样本，从本地 configs/prompts/rca.yaml 加载并格式化 RCA prompts
        ↓
subprocess 调用各 agent 的 agent_runner.py
  stdin:  { question, system_prompt, user_prompt,
            compress_system_prompt, compress_user_prompt, data_dir }
  stdout: { output (CausalGraph JSON), trajectory (OpenAI 格式) }
        ↓
结果写回 RolloutRunner 本地 DB (stage="rollout", exp_id="rollout_<agent>")
        ↓
手动 cp DB 回 RCAgentEval/
        ↓
RCAgentEval rejudge_samples.py → metrics
（⚠️ 不要用 run_eval_judge.py，会因 agent_type 不匹配创建空白新记录）
```

---

## 关键决策（已确认）

- [x] **trajectory 精度**：精确转换各框架的消息格式为 OpenAI role 格式（影响 `tool_bonus`）
- [x] **数据来源**：直接复用 RCAgentEval DB 中 stage="init" 的记录（避免重复 preprocess）
- [x] **并发数**：根据各 agent 的 API rate limit 在各自 yaml 中单独配置
- [x] **prompts 存储**：本地副本 `configs/prompts/rca.yaml`，不依赖 RCAgentEval 路径
- [x] **stdin 字段**：6 个字段——question + system_prompt + user_prompt + compress_system_prompt + compress_user_prompt + **data_dir**（从 augmented_question 用正则提取，不依赖配置文件路径）
- [x] **DB 解耦**：RolloutRunner 维护本地 DB 副本，rollout 完成后手动 cp 回 RCAgentEval
- [x] **Judge 方式**：使用 `rejudge_samples.py`（直接操作已有记录），不用 `run_eval_judge.py`
- [x] **response 格式**：agent_runner 输出前必须剥离 markdown 代码块（strip_markdown_json）
- [x] **DB 路径**：所有 UTU_DB_URL 统一用 4 斜杠绝对路径（`sqlite:////abs/path/to/db`）

---

## 目录结构

```
RolloutRunner/
├── pyproject.toml
├── configs/
│   ├── agents/
│   │   └── thinkdepthai.yaml      # 每个 agent 一个配置文件
│   └── prompts/
│       └── rca.yaml               # RCA prompts 本地副本（从 RCAgentEval 复制）
├── src/
│   ├── dataset.py                 # 从 RCAgentEval DB 读取样本
│   ├── runner.py                  # subprocess 调用 agent（asyncio + semaphore）
│   └── db_writer.py               # 结果写回 RCAgentEval DB
├── scripts/
│   └── run_rollout.py             # 入口脚本
└── docs/plans/
    ├── ROLLOUT_EVAL_PLAN.md       # 本文件
    ├── 2026-03-04-thinkdepthai-eval-design.md
    └── 2026-03-04-rolloutrunner-impl.md
```

---

## stdin/stdout 标准接口

所有 agent 的 `agent_runner.py` 遵循同一接口：

```
stdin:  JSON {
  "question":              str,   # augmented_question（含数据路径）
  "system_prompt":         str,   # RCA_ANALYSIS_SP（已 format date）
  "user_prompt":           str,   # RCA_ANALYSIS_UP（已 format incident_description）
  "compress_system_prompt": str,  # COMPRESS_FINDINGS_SP（已 format date）
  "compress_user_prompt":  str    # COMPRESS_FINDINGS_UP（已 format incident_description）
}

stdout: JSON {
  "output":     str,   # CausalGraph JSON（judge 直接解析）
  "trajectory": list   # OpenAI 格式消息列表（计算 tool_bonus）
}
```

### agent 配置文件格式

```yaml
# configs/agents/thinkdepthai.yaml
name: thinkdepthai
cmd: ["uv", "run", "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/Deep_Research
exp_id: rollout_thinkdepthai
model_name: openai:gpt-5
agent_type: thinkdepthai
concurrency: 2
timeout: 600
```

---

## 阶段一：前置确认（已完成）

- [x] 查看 RCAgentEval DB 中的 `EvaluationSample`，确认 `augmented_question` 格式
- [x] 阅读 `RCABenchProcesser.judge_one`，确认 `response`（CausalGraph JSON）和 `trajectories` 字段要求
- [x] 确认 `tool_bonus` 计算逻辑（需 OpenAI role 格式的 tool_calls/tool_call_id）
- [x] 确认 RCA prompts：`RCA_ANALYSIS_SP/UP` 用于分析，`COMPRESS_FINDINGS_SP/UP` 用于压缩输出 CausalGraph JSON

---

## 阶段二：创建 RolloutRunner 项目骨架

- [x] 创建目录结构
- [ ] 编写 `pyproject.toml`（依赖：`sqlmodel`, `pyyaml`, `sqlalchemy`, `python-dotenv`）
- [ ] 编写 `src/dataset.py`
  - 连接 RCAgentEval DB（`UTU_DB_URL` 环境变量）
  - 读取指定 `source_exp_id` 下 stage="init" 的 `EvaluationSample`
  - 返回 `SampleRecord(id, augmented_question, correct_answer, source)` 列表
- [ ] 编写 `src/runner.py`
  - 通过 stdin 传 5 字段 JSON 给 agent
  - 通过 stdout 接收 `{ output, trajectory }`
  - `_parse_last_json`：从后往前找最后一行有效 JSON（兼容调试输出）
  - asyncio subprocess，semaphore 控制并发
  - 超时处理（`asyncio.wait_for` + kill）
  - 错误处理（非零退出码/JSON 解析失败时记录 stderr，返回 None，继续其他样本）
- [ ] 编写 `src/db_writer.py`
  - 将 AgentResult 写回 EvaluationSample
  - 字段映射：`response` ← output，`trajectories` ← json.dumps(trajectory)，`stage` ← "rollout"
- [ ] 编写 `scripts/run_rollout.py`
  - 读取 agent yaml 配置
  - 从 DB 加载样本（`--source_exp_id`，默认 `rcabench_evaluation`）
  - 从 `configs/prompts/rca.yaml` 加载并格式化所有 prompts
  - 并发调用 runner，写入结果
  - 支持 `--limit N` 调试参数

---

## 阶段三：为每个 agent 添加 agent_runner.py

> **工作流程**：每个 agent 开始前，提供该项目的 CLAUDE.md，分析具体注入方案，编写 `agent_runner.py`。

### agent_runner.py 通用设计原则

- 从 stdin 读取 5 字段 payload
- **保留**原有推理框架（图拓扑、推理逻辑）
- **替换**：RCA 相关 prompt 和工具集（与 RCAgentEval 一致）
- 将结果序列化为 OpenAI 格式 trajectory 输出到 stdout

---

### thinkdepthai / Deep_Research（第一个）

详细设计见 `2026-03-04-thinkdepthai-eval-design.md`，实施计划见 `2026-03-04-rolloutrunner-impl.md`。

**Prompt 注入策略：**

| 位置 | 原始 | 替换后 |
|------|------|--------|
| `llm_call` SystemMessage | `research_agent_prompt` | `research_agent_prompt + "\n\n---\n\n" + system_prompt`（叠加） |
| 初始 HumanMessage | task_description | `user_prompt`（RCA_ANALYSIS_UP 已填入） |
| `compress_research` SystemMessage | `compress_research_system_prompt` | `compress_system_prompt`（COMPRESS_FINDINGS_SP） |
| `compress_research` HumanMessage | `compress_research_human_message` | `compress_user_prompt`（COMPRESS_FINDINGS_UP） |

**Tools：** 去掉 `tavily_search`，保留 `[think_tool, list_tables_in_directory, get_schema, query_parquet_files]`

**Output：** `compressed_research`（COMPRESS_FINDINGS_UP 驱动输出 CausalGraph JSON）

**Trajectory：** LangChain messages → OpenAI role 格式（HumanMessage/AIMessage/ToolMessage）

- [ ] 编写 `Deep_Research/agent_runner.py`（不修改原始源码）
- [ ] 冒烟测试：手动构造 payload，验证 stdout 输出格式正确

---

### Agent 2 ~ Agent N（后续）

- [ ] 提供 CLAUDE.md → 分析项目结构 → 确定注入方案 → 编写 `agent_runner.py` → 测试

---

## 阶段四：端到端验证

- [ ] 单条样本跑通完整流程：`run_rollout.py --limit 1` → DB 写入
- [ ] 检查 DB：`response` 是否为合法 CausalGraph JSON，`stage` 是否为 "rollout"
- [ ] 在 RCAgentEval 运行 judge，确认 `parse_causal_graph` 正常解析
- [ ] 确认 `tool_bonus` 计算正常（trajectory 中有 tool_calls）
- [ ] 全量跑通
- [ ] 对比多个 agent 的测评结果

---

## 运行命令参考

```bash
# 1. 跑某个 agent 的 rollout（单条调试）
cd /home/nn/SOTA-agents/RolloutRunner
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<agent>_init.db \
  uv run python scripts/run_rollout.py --agent <agent> --source_exp_id <agent>_init --limit 1

# 2. 全量 rollout
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<agent>_init.db \
  nohup uv run python scripts/run_rollout.py --agent <agent> --source_exp_id <agent>_init \
  > /tmp/rollout_<agent>.log 2>&1 &

# 3. 传回 RCAgentEval
cp /home/nn/SOTA-agents/RolloutRunner/<agent>_init.db \
   /home/nn/SOTA-agents/RCAgentEval/<agent>_init.db

# 4. 将 rollout 样本标记为 judged，再运行 rejudge
cd /home/nn/SOTA-agents/RCAgentEval
uv run python -c "
from sqlmodel import Session, select, create_engine
from utu.eval.data import EvaluationSample
engine = create_engine('sqlite:////home/nn/SOTA-agents/RCAgentEval/<agent>_init.db')
with Session(engine) as s:
    samples = list(s.exec(select(EvaluationSample).where(EvaluationSample.stage == 'rollout')).all())
    for r in samples: r.stage = 'judged'; s.add(r)
    s.commit(); print(f'Updated {len(samples)} samples')
"
uv run python scripts/rejudge_samples.py

# 5. Dashboard 可视化（需确保 .env 中 UTU_DB_URL 为绝对路径）
nohup uv run python scripts/dashboard/run_dashboard.py --mode prod --port 8001 > /tmp/dashboard.log 2>&1 &
# 本地访问：ssh -L 8001:localhost:8001 nn@10.10.10.140 → http://localhost:8001
```

> 完整流程和所有已知坑见 **EVAL_RUNBOOK.md**（同目录）
