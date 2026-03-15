# o3-deep-research 测评设计文档

> 日期：2026-03-10
> 参考：`2026-03-04-thinkdepthai-eval-design.md`（thinkdepthai 接入方案）

---

## 一、目标

在 `RolloutRunner` 项目中为 **openai/o3-deep-research**（通过 OpenRouter 调用）创建 RCA 测评接入层。
使用与 thinkdepthai 完全相同的三阶段框架（llm_call → tool_node → compress），保留 RCA 工具集（去掉 think_tool），Prompt 来自同一份 `rca.yaml`。

---

## 二、设计原则

| 原则 | 说明 |
|------|------|
| **对齐 thinkdepthai 结构** | 同样的三阶段循环：research loop → compress，stdin/stdout 接口完全一致 |
| **纯 OpenAI SDK** | 不引入 LangChain/LangGraph，使用 `openai` 包原生 function calling |
| **去掉 think_tool** | o3-deep-research 本身具备推理能力，无需外部反思工具；工具集 3 个 |
| **Prompt strip** | 运行时从 stdin 接收的 prompt 中动态去除 think_tool 引用 |
| **Tool 实现对齐** | `src/tools.py` 与 `Deep_Research/src/rca_tools.py` 完全一致（去掉 @tool 装饰器） |

---

## 三、架构对比

### 图拓扑

```
thinkdepthai (LangGraph — 图驱动):
  START → llm_call ←→ tool_node → compress_research → END
          (LangChain bind_tools, 4 tools含think_tool)

o3-deep-research (纯 Python — while 循环):
  run_research_loop():
    messages = [system, user]
    for _ in range(50):
        LLM call → if tool_calls: execute → append; else: break
  run_compress():
    messages = [compress_sp] + trajectory + [compress_up]
    LLM call → CausalGraph JSON
```

### 关键区别

| 维度 | o3-deep-research | thinkdepthai |
|------|-----------------|-------------|
| 模型 | openai/o3-deep-research (OpenRouter) | openai:kimi-k2-0905-preview |
| 框架 | 纯 Python while 循环 | LangGraph StateGraph |
| Tool 数量 | 3（无 think_tool） | 4（含 think_tool） |
| Tool 注册 | OpenAI JSON schema 列表 | LangChain `model.bind_tools()` |
| 消息格式 | 原生 OpenAI dict | LangChain Message 对象 → 最终转换为 OpenAI dict |
| Prompt 组合 | system_prompt（strip think 后直接使用） | system_prompt + rca_think_prompt 追加 |
| 迭代上限 | 50 次 LLM call | recursion_limit=100（图节点执行次数） |
| trajectory 转换 | 无需转换（已是 OpenAI dict） | convert_trajectory()（LangChain → OpenAI dict） |

---

## 四、完整数据流

```
stdin JSON (6 字段)
  │  question, system_prompt, user_prompt,
  │  compress_system_prompt, compress_user_prompt, data_dir
  ▼
main()
  │  strip_think_tool(system_prompt)       ← 删除 "4. **think_tool**..." 行
  │  strip_think_tool(compress_up)         ← 删除 "**Exclude**: think_tool..." 行
  │  user_prompt += "## Data Location\n..." + list_tables 引导
  ▼
run_research_loop(client, model, system_prompt, user_prompt)
  │  messages = [
  │    {"role": "system", "content": system_prompt},
  │    {"role": "user",   "content": user_prompt},
  │  ]
  │  for _ in range(50):
  │      response = client.chat.completions.create(
  │          model=model, messages=messages,
  │          tools=TOOLS, tool_choice="auto"
  │      )
  │      msg = response.choices[0].message
  │      messages.append(assistant_entry)    ← 含 tool_calls 或不含
  │      if not msg.tool_calls: break
  │      for tc in msg.tool_calls:
  │          result = execute_tool(tc.name, json.loads(tc.arguments))
  │          messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
  │  return messages   ← 含 system/user/assistant/tool 全部消息
  ▼
run_compress(client, model, compress_sp, compress_up, trajectory)
  │  messages = [compress_sp] + trajectory + [compress_up]
  │  response = client.chat.completions.create(model=model, messages=messages)
  │  return response.choices[0].message.content
  ▼
main()
  │  output_trajectory = [m for m in trajectory if m["role"] != "system"]
  │  result = {
  │    "output": strip_markdown_json(compressed),
  │    "trajectory": output_trajectory,
  │  }
  ▼
stdout: 单行 JSON（runner._parse_last_json 从末行解析）
```

---

## 五、Prompt 注入全景

### stdin 字段 → 注入位置

| stdin 字段 | 注入位置 | 处理 | thinkdepthai 对比 |
|-----------|---------|------|-----------------|
| `system_prompt` | `run_research_loop` → 每次 LLM call 的 system message | strip think_tool 引用后直接用 | thinkdepthai 追加 rca_think_prompt |
| `user_prompt` | `run_research_loop` → 初始 user message | 追加 data_dir 数据路径提示 | 完全一致 |
| `compress_system_prompt` | `run_compress` → system message | 直接用 | 完全一致 |
| `compress_user_prompt` | `run_compress` → 末尾 user message | strip think_tool 引用后使用 | thinkdepthai 直接用（无 strip 需要） |

### Prompt strip 规则

```python
def strip_think_tool(prompt: str) -> str:
    # RCA_ANALYSIS_SP 中的 Available Tools 第 4 条
    prompt = re.sub(r"\n?[^\n]*4\.\s+\*\*think_tool\*\*[^\n]*", "", prompt)
    # COMPRESS_FINDINGS_UP 中的 Exclude 指令
    prompt = re.sub(r"\n?[^\n]*\*\*Exclude\*\*: think_tool calls[^\n]*", "", prompt)
    return prompt
```

---

## 六、Tools 定义

### 工具列表（thinkdepthai 去掉 think_tool）

| Tool | 来源 | 功能 |
|------|------|------|
| `list_tables_in_directory` | `src/tools.py` | 递归列出 parquet 文件（ALLOWED_STEMS 过滤），返回路径/行数/列数 |
| `get_schema` | `src/tools.py` | 批量获取 schema，点列名自动重命名（`attr.x` → `attr_x`） |
| `query_parquet_files` | `src/tools.py` | DuckDB SQL 查询，自动创建 VIEW，TOKEN_LIMIT=5000 |

### 与 rca_tools.py 的对齐点

`src/tools.py` 与 `Deep_Research/src/rca_tools.py` 实现完全一致，差异仅在：
- 去掉 LangChain `@tool` 装饰器（使用纯 Python 函数）
- Tool 通过 OpenAI JSON schema 列表注册（而非 `model.bind_tools()`）

---

## 七、环境与依赖

### Python 环境

使用 `test/openai_test/` 的 conda 环境（`openai_test`，Python 3.13）：

```
openai>=1.0.0
duckdb
python-dotenv
```

无需 LangChain/LangGraph。

### 启动命令

```yaml
# RolloutRunner/configs/agents/o3_deep_research.yaml
cmd: ["python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/test/openai_test
```

> 注意：使用 conda 环境中的 `python`，不用 `uv run`。需确保 conda 环境已激活或 python 路径正确。

### 环境变量（test/openai_test/.env）

```
DEEPRESEARCH_API_KEY=sk-or-v1-...
DEEPRESEARCH_API_URL=https://openrouter.ai/api/v1
DEEPRESEARCH_MODEL=openai/o3-deep-research
```

---

## 八、RolloutRunner 配置

```yaml
# RolloutRunner/configs/agents/o3_deep_research.yaml
name: o3_deep_research
cmd: ["python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/test/openai_test
exp_id: rollout_o3_deep_research
model_name: openai/o3-deep-research
agent_type: o3_deep_research
concurrency: 1        # o3-deep-research 较慢，串行避免超时
timeout: 1800         # deep research 单样本可能需要 20-30 分钟
data_dir: /home/nn/SOTA-agents/RolloutRunner/data
```

**与 thinkdepthai 配置对比：**

| 字段 | o3_deep_research | thinkdepthai |
|------|-----------------|-------------|
| `cmd` | `["python", "agent_runner.py"]` | `["uv", "run", "python", "agent_runner.py"]` |
| `concurrency` | 1（串行） | 2 |
| `timeout` | 1800（30 min） | 600（10 min） |

---

## 九、文件清单

```
test/openai_test/
├── agent_runner.py          # RCA 评测接口（stdin/stdout，无 LangChain）
├── test_local.py            # 本地冒烟测试脚本
├── modification.md          # 本项目改造说明
├── src/
│   ├── tools.py             # DuckDB parquet 工具（对齐 rca_tools.py）
│   └── rca_agent.py         # 原始实现（已废弃，保留备查）
├── .env                     # OpenRouter API 密钥
└── data/                    # 本地测试用 parquet 数据

RolloutRunner/
├── configs/agents/o3_deep_research.yaml   # Agent 配置
└── docs/plans/2026-03-10-o3-deep-research-eval-design.md  # 本文档
```

---

## 十、运行命令

### 本地冒烟测试

```bash
cd /home/nn/SOTA-agents/test/openai_test
python test_local.py
```

### 通过 RolloutRunner 测试

```bash
cd /home/nn/SOTA-agents/RolloutRunner

# 单条冒烟
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<db>.db \
  uv run python scripts/run_rollout.py \
  --agent o3_deep_research \
  --source_exp_id <source_exp_id> \
  --limit 1

# 全量（后台）
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RolloutRunner/<db>.db \
  nohup uv run python scripts/run_rollout.py \
  --agent o3_deep_research \
  --source_exp_id <source_exp_id> \
  > /tmp/rollout_o3.log 2>&1 &

tail -f /tmp/rollout_o3.log
```

---

## 十一、已知风险与注意事项

| 风险 | 说明 | 缓解措施 |
|------|------|---------|
| 超时风险 | o3-deep-research 推理较慢，单样本可能 20-30 min | timeout=1800，concurrency=1 |
| function calling 支持 | o3-deep-research 在 OpenRouter 上支持 function calling，但行为可能与标准 chat 模型不同 | 先做 1 条冒烟测试验证 tool_calls 格式正确 |
| 无 think_tool | 模型不能调用 think_tool，但 o3 本身有推理能力，影响预计较小 | 观察 trajectory 中是否有足够的中间推理 |
| conda 环境激活 | `cmd: ["python", ...]` 依赖正确的 python 路径 | 确认 conda 环境激活，或使用绝对路径 |

---

## 十二、端到端验证步骤

1. [ ] 手动测试：`echo '<payload>' | python agent_runner.py`
2. [ ] 检查 stdout 最后一行是合法 JSON
3. [ ] 检查 `output` 是纯 CausalGraph JSON（无 markdown 包裹）
4. [ ] 检查 `trajectory` 中有 `tool_calls`（影响 tool_bonus 计算）
5. [ ] 单条 `run_rollout.py --limit 1`
6. [ ] 验证 DB 写入：`stage=rollout`，`response` 可 `json.loads()`
7. [ ] cp DB → 改 stage → rejudge → 查看 correct
8. [ ] 全量跑完 → 填充 difficulty → Dashboard 可视化
