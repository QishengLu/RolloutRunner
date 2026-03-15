# ThinkDepthAI 测评设计文档

## 目标

在 `RolloutRunner` 项目中为 Deep_Research（thinkdepthai）agent 创建测评接入层，使其能够通过 RCAgentEval 的完整 judge pipeline 进行评测。

## 设计原则

- **保留推理框架**：Deep_Research 的 LangGraph 图拓扑（START→llm_call→tool_node→compress_research→END）完全不变
- **替换 RCA 相关输入**：prompt、tools、问题格式全部与 RCAgentEval 保持一致
- **最大解耦**：RolloutRunner 通过 subprocess stdin/stdout 与 agent 交互，零代码耦合

---

## 组件一：RolloutRunner 项目

### 目录结构

```
RolloutRunner/
├── pyproject.toml
├── configs/
│   └── agents/
│       └── thinkdepthai.yaml
├── src/
│   ├── dataset.py       # 从 RCAgentEval DB 读取 stage="init" 样本
│   ├── runner.py        # asyncio subprocess + semaphore 并发控制
│   └── db_writer.py     # 结果写回 EvaluationSample
├── scripts/
│   └── run_rollout.py   # 入口脚本
└── docs/plans/
    └── 2026-03-04-thinkdepthai-eval-design.md
```

### stdin/stdout 接口

```
stdin:  JSON { "question": str, "system_prompt": str, "user_prompt": str,
               "compress_system_prompt": str, "compress_user_prompt": str, "data_dir": str }
stdout: JSON { "output": str (CausalGraph JSON), "trajectory": list }
```

### agent 配置（configs/agents/thinkdepthai.yaml）

```yaml
name: thinkdepthai
cmd: ["uv", "run", "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/Deep_Research
exp_id: rollout_thinkdepthai
model_name: openai:gpt-5
agent_type: thinkdepthai
concurrency: 2
timeout: 600
rca_prompts_path: configs/prompts/rca.yaml
```

### run_rollout.py 流程

1. 读取 agent yaml 配置
2. 从 RCAgentEval DB（`UTU_DB_URL`）加载 `stage="init"` 的 `EvaluationSample`
3. 从 `rca_prompts_path` 加载 `RCA_ANALYSIS_SP`、`RCA_ANALYSIS_UP`、`COMPRESS_FINDINGS_SP`、`COMPRESS_FINDINGS_UP`
4. 为每个样本构造 stdin payload：
   - `question` = `sample.augmented_question`
   - `system_prompt` = `RCA_ANALYSIS_SP`（不 format，由 agent_runner 自行处理 `{date}`）
   - `user_prompt` = `RCA_ANALYSIS_UP.format(incident_description=augmented_question)`
   - `compress_system_prompt` = `COMPRESS_FINDINGS_SP`
   - `compress_user_prompt` = `COMPRESS_FINDINGS_UP.format(incident_description=augmented_question)`
5. asyncio 并发调用 runner（semaphore 控制并发数）
6. 将结果写回 DB（stage="rollout"）

---

## 组件二：Deep_Research/agent_runner.py

### Prompt 注入策略

| 位置 | 原始内容 | 替换后 |
|------|---------|--------|
| `llm_call` SystemMessage | `research_agent_prompt` | `research_agent_prompt + "\n\n---\n\n" + system_prompt`（叠加，保留原有推理风格） |
| 初始 HumanMessage | `task_description`（来自 task.json） | `user_prompt`（`RCA_ANALYSIS_UP` 已填入 augmented_question） |
| `compress_research` SystemMessage | `compress_research_system_prompt` | `compress_system_prompt`（`COMPRESS_FINDINGS_SP`） |
| `compress_research` HumanMessage | `compress_research_human_message` | `compress_user_prompt`（`COMPRESS_FINDINGS_UP`） |

### Tools 替换

| 工具 | 来源 | 操作 |
|------|------|------|
| `tavily_search` | Deep_Research | **移除**（网页搜索，RCA 任务不需要） |
| `think_tool` | `Deep_Research/src/rca_tools.py` | **保留** |
| `list_tables_in_directory` | `Deep_Research/src/rca_tools.py` | **保留** |
| `get_schema` | `Deep_Research/src/rca_tools.py` | **保留** |
| `query_parquet_files` | `Deep_Research/src/rca_tools.py` | **保留** |

RCAgentEval 的 `QueryParquetFilesToolkit` 为 async，而 Deep_Research 的工具为同步 `@tool`，逻辑几乎等价（均基于 DuckDB，TOKEN_LIMIT=5000）。保留同步版本以维持图的同步调用链，无需引入 async 复杂性。

### 图构建方式

`agent_runner.py` **不修改原始源码**，而是用闭包重新参数化同样拓扑的图：

```python
from src.rca_tools import think_tool, list_tables_in_directory, get_schema, query_parquet_files

rca_tools = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
tool_node = ToolNode(rca_tools)

def make_llm_call(combined_system_prompt, tools):
    model_with_tools = init_chat_model(...).bind_tools(tools)
    def llm_call(state):
        return {"researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=combined_system_prompt)] + state["researcher_messages"]
            )
        ]}
    return llm_call

def make_compress_research(compress_sp, compress_up):
    compress_model = init_chat_model(...)
    def compress_research(state):
        messages = [SystemMessage(content=compress_sp)] \
                 + state.get("researcher_messages", []) \
                 + [HumanMessage(content=compress_up)]
        response = compress_model.invoke(messages)
        return {"compressed_research": str(response.content), "raw_notes": [...]}
    return compress_research

# 图拓扑与原始 research_agent.py 完全相同
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
agent_builder.add_node("llm_call", make_llm_call(combined_system_prompt, rca_tools))
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", make_compress_research(...))
# edges 与原始完全一致
```

### Output

- `output` = `final_state["compressed_research"]`（COMPRESS_FINDINGS 输出的 CausalGraph JSON）
- `trajectory` = LangChain messages 转 OpenAI 格式：
  - `HumanMessage` → `{"role": "user", "content": "..."}`
  - `AIMessage` → `{"role": "assistant", "content": "...", "tool_calls": [{"id": ..., "type": "function", "function": {"name": ..., "arguments": "..."}}]}`
  - `ToolMessage` → `{"role": "tool", "content": "...", "tool_call_id": "..."}`

---

## 数据库写回字段映射

| EvaluationSample 字段 | 来源 |
|----------------------|------|
| `response` | `output`（CausalGraph JSON） |
| `trajectories` | `json.dumps(trajectory)` |
| `stage` | `"rollout"` |
| `exp_id` | 配置文件中的 `exp_id` |
| `agent_type` | 配置文件中的 `agent_type` |
| `model_name` | 配置文件中的 `model_name` |
| `time_cost` | subprocess 耗时（秒） |

---

## 运行命令

```bash
# 1. 跑 rollout
cd /home/nn/SOTA-agents/RolloutRunner
uv run python scripts/run_rollout.py --agent thinkdepthai

# 2. Judge
cd /home/nn/SOTA-agents/RCAgentEval
# ⚠️ 不要用 run_eval_judge.py！
uv run python scripts/rejudge_samples.py  # 正确做法

# 3. 查看结果
uv run python scripts/rejudge_samples.py
```

---

## 端到端验证步骤

1. 单条样本跑通 `run_rollout.py`
2. 检查 DB：`response` 是否为合法 CausalGraph JSON，`stage` 是否为 `"rollout"`
3. 在 RCAgentEval 运行 judge，确认 `parse_causal_graph` 正常解析
4. 确认 `tool_bonus` 计算正常（trajectory 中有 tool_calls）
5. 全量跑通
