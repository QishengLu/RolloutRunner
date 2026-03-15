# Auto-Deep-Research 测评设计文档

## 目标

在 `RolloutRunner` 中为 Auto-Deep-Research agent 创建测评接入层，通过 RCAgentEval 完整 judge pipeline 进行 RCA 评测。

## 设计原则

- **保留 MetaChain 多 agent 架构**：Triage → Coding 双 agent 路由完全不变
- **替换 RCA 相关输入**：prompt、tools 与 RCAgentEval 保持一致
- **最大解耦**：RolloutRunner 通过 subprocess stdin/stdout 交互，零代码耦合

---

## 与 thinkdepthai 的关键差异

| 维度 | thinkdepthai | Auto-Deep-Research |
|------|-------------|-------------------|
| 框架 | LangGraph StateGraph | **MetaChain 多 agent** |
| Agent 架构 | 单 Researcher Agent | **Triage → Coding 双 agent** |
| 循环控制 | LangGraph should_continue 路由 | **MetaChain run() + case_resolved 终止** |
| 工具 | think + 3 parquet (langchain @tool) | 相同 4 工具 (原生函数 + safe_wrap) |
| LLM 接口 | langchain init_chat_model | **litellm** |
| 压缩 | LangGraph compress_research 节点 | **独立 litellm.completion 调用** |
| trajectory | LangChain Message → OpenAI 转换 | **MetaChain 原生 dict 格式** |
| 特殊处理 | 无 | **Monkey-patch litellm 支持 kimi FC** |

---

## 运行架构

### MetaChain 多 Agent 拓扑

```
stdin (6-field JSON)
  ↓
agent_runner.py main()
  ↓
build_rca_agents(system_prompt, model)
  ↓
MetaChain().run(triage_agent)  # 无轮次上限，由 case_resolved 终止
  │
  ├─ System Triage Agent
  │   ├─ instructions: RCA_ANALYSIS_SP + 调度策略
  │   ├─ tools: transfer_to_coding_agent, case_resolved, case_not_resolved
  │   └─ 决策：委派 / 继续调查 / 终结
  │
  └─ Coding Agent (via transfer_to_coding_agent)
      ├─ instructions: 数据分析专家
      ├─ tools: think_tool, list_tables, get_schema, query_parquet_files
      └─ transfer_back_to_triage_agent(findings)
  │
  ↓ case_resolved() → MetaChain 终止
compress_findings(messages, compress_sp, compress_up)
  ↓ 独立 litellm.completion 调用
strip_markdown_json() → stdout JSON
```

### Agent 切换机制

MetaChain 通过 `Result(agent=xxx)` 实现 agent 切换：
- Triage 调用 `transfer_to_coding_agent(sub_task)` → `Result(agent=coding_agent)` → MetaChain 切到 Coding
- Coding 调用 `transfer_back_to_triage_agent(findings)` → `Result(agent=triage_agent)` → 切回 Triage
- Triage 调用 `case_resolved(conclusion)` → MetaChain 退出循环

### 终止条件

1. Triage 调用 `case_resolved()` — 正常完成
2. Triage 调用 `case_not_resolved()` — 无法确定根因
3. subprocess timeout（配置 `timeout: 600`）— 外层兜底

> **不设 max_turns 上限**（MetaChain 默认 `float("inf")`），让 agent 自主决定何时终止调查。
> 由 RolloutRunner 的 subprocess timeout 防止无限运行。

---

## 组件一：agent_runner.py

**位置**: `/home/nn/SOTA-agents/Auto-Deep-Research/agent_runner.py`（已实现，304 行）

### Prompt 注入策略

| 位置 | 原始内容 | 替换后 |
|------|---------|--------|
| Triage instructions | 原始调度 prompt | `system_prompt + "\n\n---\n\n" + 调度策略`（叠加） |
| 初始 user message | — | `user_prompt + data_dir 提示`（RCA_ANALYSIS_UP） |
| compress_findings | — | `compress_sp` / `compress_up`（独立 LLM 调用） |

Coding Agent 的 instructions 保持独立，不注入 RCA prompt（它只负责数据查询）。

### Tools

| 工具 | 来源 | 安全处理 |
|------|------|---------|
| `think_tool` | `agent_runner.py` 内联定义 | 原生安全 |
| `list_tables_in_directory` | `autoagent/tools/parquet_tools.py` | `_safe_wrap` 包装 |
| `get_schema` | `autoagent/tools/parquet_tools.py` | `_safe_wrap` 包装 |
| `query_parquet_files` | `autoagent/tools/parquet_tools.py` | `_safe_wrap` 包装 |
| `transfer_to_coding_agent` | `agent_runner.py` 闭包 | Result 返回 |
| `transfer_back_to_triage_agent` | `agent_runner.py` 闭包 | Result 返回 |
| `case_resolved` | `autoagent/tools/inner.py` | 终止 MetaChain |
| `case_not_resolved` | `autoagent/tools/inner.py` | 终止 MetaChain |

> `_safe_wrap` 捕获异常返回错误字符串，防止 MetaChain crash。

### 关键技术点

1. **Monkey-patch litellm**: `litellm.supports_function_calling` 不识别 kimi 模型，
   需 patch 让含 "kimi" 的模型返回 `True`，否则 litellm 会 fallback 到非 FC 模式
2. **环境变量前置设置**: `COMPLETION_MODEL` 和 `FN_CALL` 必须在 `import constant` 之前设置，
   因为 constant.py 在模块加载时读取环境变量
3. **data_dir 追加到 user_prompt**: 明确告知 agent 数据位置和初始操作指令
4. **压缩阶段独立**: MetaChain 结束后再调用 `compress_findings()`，不在 agent 循环内

### 执行流程

```python
1. payload = json.loads(sys.stdin.read())     # 读取 6 字段
2. user_prompt += data_dir 提示               # 追加数据路径
3. triage_agent = build_rca_agents(sp, model) # 构建双 agent
4. response = MetaChain().run(triage_agent)   # 多 agent 循环
5. compressed = compress_findings(messages)   # 独立 LLM 压缩
6. output = strip_markdown_json(compressed)   # 去 markdown 包裹
7. print(json.dumps({output, trajectory}))    # stdout 输出
```

---

## 组件二：RolloutRunner 配置

### agent 配置（configs/agents/auto_deep_research.yaml）

```yaml
name: auto_deep_research
cmd: ["uv", "run", "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/Auto-Deep-Research
exp_id: rollout_auto_deep_research
model_name: kimi-k2-0905-preview
agent_type: auto_deep_research
concurrency: 2
timeout: 600
```

> `timeout: 600` — 不设 max_turns 上限，由 subprocess timeout 兜底。单样本通常 3-8 分钟，如不够可调大。
> `concurrency: 2` — 受 Moonshot API rate limit 限制，可根据实际情况调整。

### stdin/stdout 接口

与其他 agent 完全一致：

```
stdin:  JSON { "question", "system_prompt", "user_prompt",
               "compress_system_prompt", "compress_user_prompt", "data_dir" }
stdout: JSON { "output" (CausalGraph JSON), "trajectory" (OpenAI 格式) }
```

---

## 数据库写回字段映射

| EvaluationSample 字段 | 来源 |
|----------------------|------|
| `response` | `output`（CausalGraph JSON） |
| `trajectories` | `json.dumps(trajectory)` |
| `stage` | `"rollout"` |
| `exp_id` | `rollout_auto_deep_research` |
| `agent_type` | `auto_deep_research` |
| `model_name` | `kimi-k2-0905-preview` |
| `time_cost` | subprocess 耗时（秒） |

---

## 环境依赖

### Auto-Deep-Research 项目环境

```bash
cd /home/nn/SOTA-agents/Auto-Deep-Research
uv venv && uv sync    # Python >=3.10
```

### .env 配置

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_BASE=https://api.moonshot.cn/v1
RCA_MODEL=openai/kimi-k2-0905-preview
```

### 关键依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `litellm` | 1.55.0 | 多 LLM 统一接口 + function calling |
| `openai` | >=1.52.0 | OpenAI SDK |
| `duckdb` | — | Parquet SQL 查询 |
| `python-dotenv` | — | .env 加载 |
| `pydantic` | — | 类型验证 |

---

## 运行命令

```bash
# === 冒烟测试（1 条） ===
cd /home/nn/SOTA-agents/RolloutRunner
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/auto_deep_research-kimi-k2.db \
  uv run python scripts/run_rollout.py \
    --agent auto_deep_research \
    --source_exp_id <init_exp_id> \
    --limit 1

# === 全量运行 ===
nohup uv run python -u scripts/run_rollout.py \
  --agent auto_deep_research \
  --source_exp_id <init_exp_id> \
  > /tmp/rollout_auto_deep_research.log 2>&1 &

# === 查看进度 ===
sqlite3 auto_deep_research-kimi-k2.db \
  "SELECT stage, COUNT(*) FROM evaluation_data GROUP BY stage"
tail -f /tmp/rollout_auto_deep_research.log

# === Judge（评测完成后） ===
cd /home/nn/SOTA-agents/RCAgentEval
cp /home/nn/SOTA-agents/RolloutRunner/auto_deep_research-kimi-k2.db ./

# 改 stage（rejudge 查 stage="judged"，不是 "rollout"）
sqlite3 auto_deep_research-kimi-k2.db \
  "UPDATE evaluation_data SET stage='judged' WHERE stage='rollout'"

# 填充 difficulty 元数据（Dashboard 分布图必需）
# 见 EVAL_RUNBOOK.md Step 7.5

# 运行 rejudge
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RCAgentEval/auto_deep_research-kimi-k2.db \
  uv run python scripts/rejudge_samples.py

# 查看结果
sqlite3 auto_deep_research-kimi-k2.db \
  "SELECT correct, COUNT(*) FROM evaluation_data WHERE stage='judged' GROUP BY correct"
```

---

## 已知风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| litellm monkey-patch 失效 | kimi FC 不可用，agent 降级为纯文本 | 检查 litellm 版本兼容性，patch 在 import 后立即执行 |
| subprocess timeout 触发 | 调查被截断 | timeout=600 兜底，需确认是否足够；可按需调大 |
| Coding Agent 死循环 | 不断查询不 transfer_back | subprocess timeout 兜底 |
| case_not_resolved | 无法确定根因 | compress_findings 仍会被调用，输出部分结果 |
| parquet 工具异常 | 查询失败 | `_safe_wrap` 返回错误字符串，agent 可重试或换策略 |
| compress 输出含 markdown | judge 解析失败 | `strip_markdown_json()` 已处理 |
| 双 agent trajectory 混杂 | tool_bonus 计算可能不同 | `extract_trajectory` 统一为 OpenAI 格式，含 sender 信息 |

---

## 端到端验证步骤

1. 单条样本跑通 `run_rollout.py --limit 1`
2. 检查 DB：
   - `response` 是否为合法 CausalGraph JSON（含 nodes, edges, root_causes）
   - `stage` 是否为 `"rollout"`
   - `trajectories` 是否包含 assistant/tool 消息
3. 检查 trajectory 中：
   - Triage Agent 和 Coding Agent 的消息是否交替出现
   - tool_calls 中是否包含 parquet 工具调用
   - 是否有 `case_resolved` 终止调用
4. 在 RCAgentEval 运行 rejudge，确认：
   - `parse_causal_graph` 正常解析 response
   - `tool_bonus` 计算正常（trajectory 中有 tool_calls）
   - graph_metrics 正常写入 `meta.graph_metrics.primary.*`
5. 全量跑通（100 样本），比较与 thinkdepthai 的 AC@1 差异

---

## 预期对比分析

Auto-Deep-Research 与 thinkdepthai 使用相同的模型（kimi-k2）和 prompt（rca.yaml），
主要差异在于 agent 架构：

- **双 agent 架构**可能带来更好的任务分解（Triage 负责策略，Coding 负责执行）
- 无轮次上限意味着 agent 可以充分调查，但也可能因 handoff 开销导致耗时更长
- `tool_choice="required"` 确保 Coding Agent 每轮必调工具，避免空转
- `parallel_tool_calls=False` 保证顺序执行，与 thinkdepthai 行为一致

关注指标：
- **AC@1**: 与 thinkdepthai 43.6% 对比
- **平均 tool 调用次数**: 无轮次限制下 agent 是否做更充分的调查
- **平均耗时**: 无上限轮次 + compress 独立调用的总时间分布
- **timeout 率**: 600s timeout 是否足够，是否需要调大
