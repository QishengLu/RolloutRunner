# AIQ (NVIDIA AIRA) 测评设计文档

> 日期：2026-03-10
> 参考：`2026-03-04-thinkdepthai-eval-design.md`（thinkdepthai 接入方案）

---

## 一、目标

在 `RolloutRunner` 项目中为 AIQ（NVIDIA AIRA）agent 创建 RCA 测评接入层。保留 AIRA 原始多阶段 LangGraph 流水线架构，仅将 RAG/Tavily Web 搜索替换为 DuckDB Parquet 工具。

---

## 二、设计原则

| 原则 | 说明 |
|------|------|
| **保留原始框架** | AIRA 的 5 节点线性流水线（generate_queries → data_research → summarize → reflect → finalize）完全保留，不替换为 thinkdepthai 的单循环架构 |
| **替换搜索源** | RAG + Tavily Web Search → DuckDB Parquet 工具（与 thinkdepthai 使用相同的 4 个工具） |
| **注入 RCA 评测输入** | 通过 RolloutRunner stdin 传入 system_prompt、user_prompt、compress prompts，分别注入对应节点 |
| **不修改原项目源码** | `agent_runner.py` 放在 `aiq/` 根目录，不 import AIRA 原始包（避免 nvidia-nat 编译问题） |
| **最大解耦** | RolloutRunner 通过 subprocess stdin/stdout 与 agent 交互，零代码耦合 |

---

## 三、AIQ vs thinkdepthai 架构对比

### 图拓扑

```
AIQ (AIRA 多阶段流水线 — 编排驱动):
  START → generate_queries → data_research → summarize_sources → reflect_on_summary → finalize_summary → END
         （纯 LLM）      （工具调用循环）    （纯 LLM）        （LLM + 工具循环）       （纯 LLM）

thinkdepthai (单循环 — LLM 驱动):
  START → llm_call ←→ tool_node → compress_research → END
         （LLM 自主决定何时查询、何时停止）
```

### 关键区别

| 维度 | AIQ | thinkdepthai |
|------|-----|-------------|
| 架构 | 5 节点线性流水线，编排驱动 | 单循环，LLM 驱动 |
| 节点数 | 5 个专用节点 | 3 个通用节点 |
| LLM 调用总数 | ~40-100 次（分散在多节点） | ~10-30 次（集中在一个循环） |
| 上下文共享 | 节点间通过 state 传递摘要，工具调用上下文不跨 query | 全部对话在一个 messages 列表中累积 |
| system_prompt 注入 | 仅影响工具调用循环（Node 2, 4） | 贯穿所有 LLM 调用 |
| 反思机制 | 图级节点 `reflect_on_summary`（结构化反思） | 无专门反思（LLM 自主判断） |
| 最终 JSON 输入 | 压缩后的 running_summary（已摘要化） | 完整对话历史（含原始工具输出） |
| 工具集 | 完全一致（4 个） | 完全一致（4 个） |

---

## 四、AIRA 原始节点 → RCA 适配映射

| AIRA 原始节点 | RCA 适配节点 | 原始实现 | 适配变化 |
|--------------|-------------|---------|---------|
| `generate_query` (Stage 1) | `generate_queries` | `nodes.py:41` + `query_writer_instructions` | prompt 改为 `RCA_QUERY_WRITER`，去除 Nemotron `</think>` 依赖 |
| `web_research` (Stage 2) | `data_research` | `nodes.py:114` + `process_single_query` (RAG+Tavily) | 替换为 `run_data_exploration()` (parquet 工具循环) |
| `summarize_sources` | `summarize_sources` | `nodes.py:168` + `summarize_report()` helper | 内联实现，prompt 改为 `RCA_SUMMARIZER`/`RCA_REPORT_EXTENDER` |
| `reflect_on_summary` | `reflect_on_summary` | `nodes.py:199` + `reflection_instructions` + `process_single_query` | prompt 改为 `RCA_REFLECTION`，搜索改为 `run_data_exploration()` |
| `finalize_summary` | `finalize_summary` | `nodes.py:315` + `finalize_report` + citations 拼接 | 使用 RolloutRunner 的 compress prompts，输出 CausalGraph JSON |

### 移除的 AIRA 组件

| 组件 | 原始用途 | 移除原因 |
|------|---------|---------|
| `search_utils.py::process_single_query()` | RAG + relevancy check + Tavily fallback | 替换为 `run_data_exploration()` |
| `search_utils.py::deduplicate_and_format_sources()` | 5 参数版（含 citations, relevancy） | 简化为 2 参数版（queries + findings） |
| `report_gen_utils.py::summarize_report()` | async 流式摘要 | 内联同步实现 |
| `tools.py` | RAG + Tavily 工具定义 | 替换为 `rca_tools.py` |
| `state.citations` | Web 引用管理 | RCA 无需引用 |
| `relevancy_checker` prompt | RAG 相关性判断 | Parquet 查询不需要相关性筛选 |
| `update_system_prompt()` | Nemotron "detailed thinking on" | 不使用 Nemotron 模型 |
| `StreamWriter` | 前端流式 UI | RolloutRunner 不需要流式输出 |

---

## 五、完整数据流

```
stdin JSON (6 字段)
  │
  ▼
main() — 解析 payload
  │  question = augmented_question + data_dir 增强
  │    （与 thinkdepthai 一致：追加 "## Data Location" + list_tables 引导）
  │  config["configurable"] = {
  │    question,            ← augmented question
  │    user_prompt,         ← RCA_ANALYSIS_UP（已 format incident_description）
  │    system_prompt,       ← RCA_ANALYSIS_SP（已 format date）
  │    data_dir,            ← parquet 数据目录
  │    compress_system_prompt, compress_user_prompt,
  │    number_of_queries=5, num_reflections=1
  │  }
  │
  ▼ agent.invoke(initial_state, config)
  │
  ├─► Node 1: generate_queries
  │     输入: config.question (augmented)
  │     Prompt: RCA_QUERY_WRITER.format(incident=question, number_of_queries=5)
  │     LLM: 1 次调用（无工具）
  │     解析: 正则 \[.*\] 提取 JSON 列表
  │     输出: state.queries = [{query, report_section, rationale}, ...] × 5
  │
  ├─► Node 2: data_research
  │     输入: state.queries, config.data_dir, config.system_prompt
  │     循环: 每个 query → run_data_exploration()
  │       │  SystemMessage(system_prompt + "---" + DATA_RESEARCH_SP)
  │       │  HumanMessage("Investigation query: {query}\n\nData location: {data_dir}")
  │       │  → LLM + bind_tools(RCA_TOOLS) → 工具调用循环（最多 30 轮）
  │       │     典型链: list_tables → get_schema → query_parquet × N → think_tool → ...
  │       │  → findings 文本 + tool_messages 列表
  │     格式化: deduplicate_and_format_sources() → XML <sources>
  │     输出: state.data_research_results = [XML],
  │           state.all_tool_messages += 所有工具消息
  │
  ├─► Node 3: summarize_sources
  │     输入: state.data_research_results[-1], state.running_summary
  │     Prompt: 首次 → RCA_SUMMARIZER.format(sources=XML)
  │             已有 → RCA_REPORT_EXTENDER.format(report=summary, source=XML)
  │     LLM: 1 次调用（无工具）
  │     输出: state.running_summary = "RCA 分析报告文本"
  │
  ├─► Node 4: reflect_on_summary (循环 num_reflections=1 次)
  │     Step 1: RCA_REFLECTION.format(report=summary, topic=question) → LLM → follow-up query
  │     Step 2: run_data_exploration(follow_up_query) → 补充数据（工具调用循环）
  │     Step 3: deduplicate_and_format_sources() → 追加到 data_research_results
  │     Step 4: RCA_REPORT_EXTENDER.format(report=summary, source=new_xml) → LLM → 扩展分析
  │     输出: state.running_summary（更新）, state.all_tool_messages += 反思轮工具消息
  │
  ├─► Node 5: finalize_summary
  │     输入: state.running_summary, config.compress_system_prompt, config.compress_user_prompt
  │     Messages: SystemMessage(compress_sp) + HumanMessage(summary + compress_up)
  │     LLM: 1 次调用（无工具），max_tokens=32000
  │     输出: state.final_report = CausalGraph JSON 字符串
  │
  ▼
main() — 输出
  output = strip_markdown_json(final_report)
  trajectory = convert_trajectory(all_tool_messages)  ← 转 OpenAI 格式
  stdout: {"output": "{\"nodes\":[...],\"edges\":[...],\"root_causes\":[...]}", "trajectory": [...]}
```

---

## 六、Prompt 注入全景

### stdin 字段 → 注入位置

| stdin 字段 | 注入节点 | 注入方式 | thinkdepthai 对比 |
|-----------|---------|---------|-----------------|
| `question` | `generate_queries` | `RCA_QUERY_WRITER.format(incident=question)` 的 HumanMessage | thinkdepthai 不使用 question |
| | `reflect_on_summary` | `RCA_REFLECTION.format(topic=question)` 的 HumanMessage | |
| `system_prompt` | `data_research` → `run_data_exploration()` | `SystemMessage(system_prompt + "---" + DATA_RESEARCH_SP)` | thinkdepthai: `SystemMessage(system_prompt + "---" + rca_think_prompt)` |
| | `reflect_on_summary` → `run_data_exploration()` | 同上 | |
| `user_prompt` | `generate_queries` (fallback) | 当 question 为空时使用 | thinkdepthai: `HumanMessage(user_prompt)` 作为初始消息 |
| `compress_system_prompt` | `finalize_summary` | `SystemMessage(compress_sp)` | thinkdepthai: `SystemMessage(compress_sp)` — 完全一致 |
| `compress_user_prompt` | `finalize_summary` | `HumanMessage` 的一部分 | thinkdepthai: `HumanMessage(compress_up)` — 完全一致 |

### 自定义 Prompt 对照表

| agent_runner.py Prompt | 对应 AIRA 原始 prompt | 注入节点 | 用途 |
|------------------------|---------------------|---------|------|
| `RCA_QUERY_WRITER` | `query_writer_instructions` | Node 1: generate_queries | 事件描述 → 5 个调查子查询 |
| `DATA_RESEARCH_SP` | 新增（替代 RAG/Tavily） | Node 2/4: run_data_exploration | 工具使用系统提示 |
| `RCA_SUMMARIZER` | `summarizer_instructions` | Node 3: summarize_sources | 首次汇总数据发现 |
| `RCA_REPORT_EXTENDER` | `report_extender` | Node 3/4: summarize/reflect | 扩展已有分析 |
| `RCA_REFLECTION` | `reflection_instructions` | Node 4: reflect_on_summary | 发现知识缺口 |
| (stdin) `compress_*` | `finalize_report` | Node 5: finalize_summary | 生成 CausalGraph JSON |

---

## 七、Tools 定义

### 工具列表（与 thinkdepthai 完全一致）

```python
RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
```

| Tool | 来源 | 功能 | 调用位置 |
|------|------|------|---------|
| `think_tool` | `agent_runner.py` 内联 | 反思占位（返回记录的思考） | `run_data_exploration()` 内部 |
| `list_tables_in_directory` | `rca_tools.py` | 列出 parquet 文件 + 元数据 | `run_data_exploration()` 内部 |
| `get_schema` | `rca_tools.py` | 获取 parquet schema | `run_data_exploration()` 内部 |
| `query_parquet_files` | `rca_tools.py` | DuckDB SQL 查询，TOKEN_LIMIT=5000 | `run_data_exploration()` 内部 |

### 工具调用逻辑

工具只在 `run_data_exploration()` 内部被调用（与 thinkdepthai 的图级 `tool_node` 不同）。

```
run_data_exploration(query, data_dir, system_prompt)
  │
  SystemMessage(system_prompt + "---" + DATA_RESEARCH_SP)
  HumanMessage("Investigation query: {query}\n\nData location: {data_dir}\n...")
  │
  for _ in range(30):  # max iterations
      response = model_with_tools.invoke(messages)
      if not response.tool_calls:
          break  ← LLM 判断信息充足，返回 findings
      for tc in response.tool_calls:
          result = RCA_TOOLS_BY_NAME[tc.name].invoke(tc.args)
          messages.append(ToolMessage(content=result, ...))
  │
  return (findings_text, all_tool_messages)
```

**调用方：**

| 调用方 | 次数 | 场景 |
|-------|------|------|
| `data_research` (Node 2) | 5 次（每个 query 1 次） | 初始数据探索 |
| `reflect_on_summary` (Node 4) | 1 次/轮 × num_reflections(1) | 补充调查 |
| **总计** | 6 次独立的 mini-agent 循环 | |

**典型单次工具调用链：**

```
Round 1: list_tables_in_directory(data_dir)  → 文件列表
Round 2: get_schema(file1), get_schema(file2)  → 列定义
Round 3: query_parquet_files("SELECT ...")  → 数据结果
Round 4: think_tool("发现 service-A 错误率异常...")  → 反思
Round 5-N: query_parquet_files("SELECT ...") × N  → 更多数据
Round N+1: AIMessage(content="Key findings: ...")  → 无 tool_calls，循环终止
```

---

## 八、State 设计

```python
class RCAState(TypedDict):
    queries: list[dict]                           # 调查查询列表
    data_research_results: list[str]              # XML <sources> 格式的数据发现
    running_summary: str                          # 运行中的 RCA 分析报告
    final_report: str                             # 最终 CausalGraph JSON
    all_tool_messages: Annotated[list, operator.add]  # 工具调用轨迹（跨节点累积）
```

**对比 AIRA 原始 AIRAState：**

| AIRAState 字段 | RCAState 字段 | 变化 |
|---------------|-------------|------|
| `queries: list[GeneratedQuery]` | `queries: list[dict]` | Pydantic → dict（简化） |
| `web_research_results: list[str]` | `data_research_results: list[str]` | 重命名 |
| `citations: str` | — | 删除（RCA 无引用） |
| `running_summary: str` | `running_summary: str` | 保留 |
| `final_report: str` | `final_report: str` | 保留 |
| — | `all_tool_messages: Annotated[list, operator.add]` | 新增（trajectory 累积） |

---

## 九、环境与依赖

### 问题：nvidia-nat 编译失败

AIRA 原始 `pyproject.toml` 依赖 `nvidia-nat[langchain,opentelemetry,weave]~=1.2`，其传递依赖链：

```
nvidia-nat → presidio-analyzer → spacy → thinc → C++ 编译
```

在 Python 3.13 上 `thinc` 的 C++ 扩展编译失败。

### 解决方案：`uv run --no-project`

绕过项目依赖，手动指定最小依赖集：

```bash
uv run --no-project --python 3.12 \
  --with "langgraph>=0.2.69,langchain-openai,langchain-core,langchain,duckdb,python-dotenv" \
  python agent_runner.py
```

### 环境变量

`aiq/.env` 文件：
```
OPENAI_API_KEY=sk-...              # kimi-k2 API 密钥
OPENAI_BASE_URL=https://api.moonshot.cn/v1
```

---

## 十、RolloutRunner 配置

### Agent 配置文件

```yaml
# RolloutRunner/configs/agents/aiq.yaml
name: aiq
cmd: ["uv", "run", "--no-project", "--python", "3.12",
      "--with", "langgraph>=0.2.69,langchain-openai,langchain-core,langchain,duckdb,python-dotenv",
      "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/aiq
exp_id: rollout_aiq
model_name: kimi-k2-0905-preview
agent_type: aiq
concurrency: 2
timeout: 600
data_dir: /home/nn/SOTA-agents/RolloutRunner/data
```

**与 thinkdepthai 配置对比：**

| 字段 | aiq | thinkdepthai |
|------|-----|-------------|
| `cmd` | `uv run --no-project --python 3.12 --with "..."` | `uv run python agent_runner.py` |
| `cwd` | `/home/nn/SOTA-agents/aiq` | `/home/nn/SOTA-agents/Deep_Research` |
| `exp_id` | `rollout_aiq` | `rollout_thinkdepthai` |
| `agent_type` | `aiq` | `thinkdepthai` |
| `timeout` | 600（多阶段流水线耗时更长） | 600 |

**cmd 差异原因**：aiq 使用 `--no-project` 绕过 nvidia-nat 依赖编译问题，thinkdepthai 项目可以直接 `uv run`。

---

## 十一、数据库写回字段映射

| EvaluationSample 字段 | 来源 |
|----------------------|------|
| `response` | `output`（CausalGraph JSON，已 strip_markdown_json） |
| `trajectories` | `json.dumps(trajectory)`（OpenAI role 格式） |
| `stage` | `"rollout"` |
| `exp_id` | `rollout_aiq`（配置文件） |
| `agent_type` | `aiq`（配置文件） |
| `model_name` | `kimi-k2-0905-preview`（配置文件） |
| `time_cost` | subprocess 耗时（秒） |

---

## 十二、文件清单

```
aiq/                                    # AIQ 项目根目录
├── agent_runner.py                     # RCA 评测接口（新增，686 行）
├── rca_tools.py                        # DuckDB parquet 工具（从 Deep_Research 复制）
├── .env                                # API 密钥（kimi-k2）
├── CLAUDE.md                           # 项目文档
│
├── aira/src/aiq_aira/                  # 原始 AIRA 包（不修改，仅参考）
│   ├── nodes.py                        # 参考：原始 5 节点实现
│   ├── schema.py                       # 参考：AIRAState 定义
│   ├── prompts.py                      # 参考：原始 prompt 模板
│   ├── search_utils.py                 # 参考：process_single_query（已被替代）
│   ├── report_gen_utils.py             # 参考：summarize_report（已内联）
│   └── ...
│
RolloutRunner/
├── configs/agents/aiq.yaml             # Agent 配置
├── configs/prompts/rca.yaml            # RCA Prompts（共用）
└── docs/plans/2026-03-10-aiq-eval-design.md  # 本文档
```

---

## 十三、运行命令

### 冒烟测试

```bash
cd /home/nn/SOTA-agents/RolloutRunner

# 单条测试
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<db>.db \
  uv run python scripts/run_rollout.py \
  --agent aiq \
  --source_exp_id <source_exp_id> \
  --limit 1
```

### 全量运行

```bash
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<db>.db \
  nohup uv run python scripts/run_rollout.py \
  --agent aiq \
  --source_exp_id <source_exp_id> \
  > /tmp/rollout_aiq.log 2>&1 &

# 监控
tail -f /tmp/rollout_aiq.log
```

### 评分流程

```bash
cd /home/nn/SOTA-agents/RCAgentEval

# 1. cp DB
cp /home/nn/SOTA-agents/RolloutRunner/<db>.db ./<db>.db

# 2. 改 stage
sqlite3 <db>.db "UPDATE evaluation_data SET stage='judged' WHERE stage='rollout'"

# 3. 填充 difficulty（参考 EVAL_RUNBOOK Step 7.5）

# 4. rejudge
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RCAgentEval/<db>.db \
  uv run python scripts/rejudge_samples.py

# 5. 查看结果
sqlite3 <db>.db "SELECT correct, COUNT(*) FROM evaluation_data WHERE stage='judged' GROUP BY correct"
```

---

## 十四、已知风险与注意事项

| 风险 | 说明 | 缓解措施 |
|------|------|---------|
| LLM 调用次数多 | 5 个 query × ~15 轮/query + 反思 ≈ 80+ 次 API 调用/样本 | 可调 `number_of_queries`(5→3) 和 `num_reflections`(1→0) |
| 上下文不跨 query | 每次 `run_data_exploration` 新建 messages，第 3 个 query 看不到第 1 个 query 的发现 | 设计如此（AIRA 原架构），通过 summarize 节点汇总 |
| timeout 风险 | 多阶段流水线总耗时可能超 600s | 必要时增大 `timeout` 到 900 或 1200 |
| system_prompt 只影响部分节点 | Node 1(generate_queries) 和 Node 3(summarize) 不注入 system_prompt | AIRA 原始设计如此，各节点有专用 prompt |
| think_tool docstring 较短 | 相比 thinkdepthai 的 `rca_think_prompt` 缺少详细反思框架 | AIRA 用图级 `reflect_on_summary` 节点做结构化反思，不依赖 tool docstring |

---

## 十五、端到端验证步骤

1. [ ] 手动测试 agent_runner.py：`echo '<payload>' | uv run --no-project ... python agent_runner.py`
2. [ ] 检查 stdout 最后一行是合法 JSON
3. [ ] 检查 output 是纯 CausalGraph JSON（无 markdown 包裹）
4. [ ] 检查 trajectory 含 tool_calls（影响 tool_bonus）
5. [ ] 单条 `run_rollout.py --limit 1`
6. [ ] 验证 DB 写入：`stage=rollout`，response 可 `json.loads()`
7. [ ] cp DB → 改 stage → rejudge → 查看 correct 结果
8. [ ] 全量跑完 → 填充 difficulty → Dashboard 可视化
