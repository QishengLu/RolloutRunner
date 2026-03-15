# RCA Agent 测评完整 Runbook

> 基于 thinkdepthai (Deep_Research) 测评实践总结，适用于所有后续 agent 的接入。

---

## 一、整体数据流

### 模式 A：PostgreSQL 多 Agent 共享 DB（推荐）

适用于多 agent 批量评测，支持 9×5=45 并发写入。

```
[服务器] docker compose up -d
      ↓  启动 PostgreSQL (port 5433, DB: SOTA-Agents)

[RCAgentEval] process_rcabench.py
      ↓  生成 DatasetSample（data 表，共享 500 行）

[RCAgentEval] preprocess_only.py --exp_id <agent>-<model>
      ↓  为每个 agent 生成 EvaluationSample (stage="init")
      ↓  exp_id 命名: thinkdepthai-kimi-k2, openrca-gpt4 等

[RolloutRunner] run_rollout.py --agent <name> --source_exp_id <exp_id>
      ↓  9 个 agent 可同时并发，各自按 exp_id 互不干扰
      ↓  中断后重跑自动跳过已完成的（load_samples 只取 stage="init"）

[RCAgentEval] rejudge_samples.py --exp_id <exp_id>
      ↓  直接查 stage="rollout"+"judged"，无需手动改 stage
      ↓  评分 + graph_metrics → stage="judged"

[RCAgentEval] Dashboard (port 8001)
      ↓  多 agent 横向对比可视化
```

**关键优势**：
- 无需 cp DB 文件（所有组件连同一个 PostgreSQL）
- 无需手动改 stage（rejudge 自动处理 rollout + judged）
- 支持高并发写入（PostgreSQL 行级锁 vs SQLite 文件锁）
- DataGrip 可从本地直接连接查询

### 模式 B：SQLite 单 Agent（旧模式，已废弃）

```
preprocess → cp .db → rollout → cp .db 回来 → rejudge → dashboard
```

---

## 二、Step-by-Step 操作手册（PostgreSQL 多 Agent 模式）

### Step 0：启动 PostgreSQL

```bash
cd /home/nn/SOTA-agents/RCAgentEval/docker
docker compose up -d

# 验证连接
psql -h localhost -p 5433 -U postgres -d SOTA-Agents -c "SELECT 1"

# 所有后续步骤统一环境变量
export UTU_DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
```

**DataGrip 远程连接**（本地电脑）：
- 方式 1：SSH Tunnel → `ssh -L 5433:localhost:5433 nn@server`，然后 DataGrip 连 `localhost:5433`
- 方式 2：DataGrip 数据源配置 → SSH/SSL tab → 勾选 Use SSH tunnel

---

### Step 1：生成数据集 + 初始化评测样本

```bash
cd /home/nn/SOTA-agents/RCAgentEval

# 生成 DatasetSample（data 表，所有 agent 共享，只需运行一次）
uv run python scripts/process_rcabench.py data --sample-size 500

# 为每个 agent+model 组合生成 EvaluationSample（stage="init"）
uv run python scripts/preprocess_only.py --exp_id thinkdepthai-kimi-k2
uv run python scripts/preprocess_only.py --exp_id openrca-gpt4
uv run python scripts/preprocess_only.py --exp_id deepresearch-claude
# ...每个 agent 一条命令，exp_id 命名: <agent_name>-<model_name>
```

**⚠️ 注意事项：**
- preprocess 会在 `eval-data/<exp_id>/data_XXXXXXXX/` 创建随机命名的 symlink，**路径已嵌入 augmented_question**
- `--sample-size` 控制样本数，默认全量
- data 表只需生成一次，多次 preprocess 不会重复插入

---

### Step 3：为新 Agent 编写 agent_runner.py

参考 `Deep_Research/agent_runner.py`，每个 agent 需要实现：

**stdin 接口（6 字段）：**
```json
{
  "question":               "augmented_question 原文",
  "system_prompt":          "RCA_ANALYSIS_SP（已 format date）",
  "user_prompt":            "RCA_ANALYSIS_UP（已 format incident_description）",
  "compress_system_prompt": "COMPRESS_FINDINGS_SP（已 format date）",
  "compress_user_prompt":   "COMPRESS_FINDINGS_UP（已 format date+incident_description）",
  "data_dir":               "/path/to/eval-data/<exp_id>/data_XXXXXXXX"
}
```

**stdout 接口（必须是最后一行 JSON）：**
```json
{
  "output": "{\"nodes\": [...], \"edges\": [...], \"root_causes\": [...]}",
  "trajectory": [
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."}
  ],
  "usage": {"total_tokens": 12345, "prompt_tokens": 8000, "completion_tokens": 4345, "llm_call_count": 8}
}
```

**关键实现点：**

| 要求 | 说明 |
|------|------|
| `output` 必须是纯 JSON | **不能有 markdown 代码块**（` ```json ``` `）！judge 直接 `json.loads()` |
| trajectory 格式 | 必须是 OpenAI role 格式（assistant/tool/user），影响 `tool_bonus` 计算 |
| `usage` 字段 | 可选，由 `UsageTracker` 自动采集实际 token 用量，写入 `meta.cost_metrics` |
| `data_dir` 使用 | 告诉 agent 去哪个目录读取 parquet 文件，从 payload 的 `data_dir` 字段读取 |
| 退出码 | 成功必须 exit 0，否则 runner 会记为失败 |
| 不修改原项目源码 | agent_runner.py 放在 agent 项目根目录，通过 closure/参数注入替换组件 |

---

### Step 4：添加 Agent 配置文件

```yaml
# RolloutRunner/configs/agents/<agentname>.yaml
name: <agentname>
cmd: ["uv", "run", "python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/<AgentProject>
exp_id: rollout_<agentname>
model_name: <model-id>          # 用于写入 DB，供 dashboard 展示
agent_type: <agentname>         # 用于写入 DB，供 dashboard 过滤
concurrency: 2                  # 并发数，根据 API rate limit 调整
timeout: 600                    # 单个样本超时秒数
data_dir: /home/nn/SOTA-agents/RolloutRunner/data  # 备用，实际由 augmented_question 提取
```

---

### Step 5：运行 Rollout

```bash
cd /home/nn/SOTA-agents/RolloutRunner

# 单条冒烟测试（先验证 agent_runner.py 能正常工作）
uv run python scripts/run_rollout.py --agent <agentname> --source_exp_id <agentname>-claude-sonnet-4.6 --limit 1
```

#### 推荐方式：串行 + tmux 持久化（2026-03-14 起）

```bash
# 一次性启动：在 tmux 中串行跑所有 8 个 agent（一个跑完再跑下一个）
bash /home/nn/SOTA-agents/RolloutRunner/launch_eval.sh

# 断线后恢复监控
tmux attach -t eval

# 查看某 agent 实时日志（在 runner window 中）
# 或
tail -f /home/nn/SOTA-agents/RolloutRunner/logs/<name>-4.6.log
```

**执行顺序**（`run_eval_sequential.sh`）：
thinkdepthai → deerflow → auto_deep_research → deepresearchagent → aiq → taskweaver → openrca → mabc

**AIMD 并发控制**：每个 agent 内部从 capacity=1 慢启动，每连续 10 次成功 +1（最高 5），失败后减半并指数退避。不需要手动调并发。

#### 旧方式（不推荐，SSH 断开即终止）

```bash
nohup uv run python -u scripts/run_rollout.py --agent thinkdepthai --source_exp_id thinkdepthai-claude-sonnet-4.6 > log_thinkdepthai.log 2>&1 &
```

```bash
# 查看所有 agent 进度（PostgreSQL）
psql -h localhost -p 5433 -U postgres -d SOTA-Agents \
  -c "SELECT exp_id, stage, COUNT(*) FROM evaluation_data GROUP BY exp_id, stage ORDER BY exp_id"
```

**断点恢复**：中断后直接重跑同一命令即可，`load_samples` 只取 `stage="init"` 的样本，已完成的自动跳过。

---

### Step 6：评分（Judge）

```bash
cd /home/nn/SOTA-agents/RCAgentEval

# ⚠️ 不要用 run_eval_judge.py！（会因 agent_type 不匹配创建空白新记录）
# ✅ 改用 rejudge_samples.py --exp_id

# 直接 rejudge，无需手动改 stage（自动查 stage="rollout" + "judged"）
uv run python scripts/rejudge_samples.py --exp_id thinkdepthai-kimi-k2
uv run python scripts/rejudge_samples.py --exp_id openrca-gpt4
# ...每个 agent 一条命令
# ⚠️ 共享 DB 时必须指定 --exp_id，否则会重新 judge 所有 agent 的数据
```

查看结果：
```bash
# 所有 agent AC@1 横向对比
psql -h localhost -p 5433 -U postgres -d SOTA-Agents -c "
SELECT agent_type, model_name,
       COUNT(*) as total,
       SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_count,
       ROUND(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) * 100, 1) as ac_at_1
FROM evaluation_data WHERE stage = 'judged'
GROUP BY agent_type, model_name ORDER BY ac_at_1 DESC"

# 按 fault_category 对比各 agent
psql -h localhost -p 5433 -U postgres -d SOTA-Agents -c "
SELECT agent_type,
       meta->'difficulty'->>'fault_category' as fault_cat,
       COUNT(*) as total,
       SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_count
FROM evaluation_data WHERE stage = 'judged'
GROUP BY agent_type, fault_cat ORDER BY agent_type, fault_cat"
```

---

### Step 7.5：填充 Difficulty 元数据（Dashboard 分布图必需）

Dashboard 的 Distribution 图表（by Fault Category / SPL / N_svc / N_edge / Fault Type / Root Cause Service）需要 `data.meta.difficulty` 预填充。

**元数据来源：**
- `causal_graph.json`：计算 SPL（最短路径长度）、n_svc（唯一服务数）、n_edge（边数）
- `injection.json`：获取 fault_type index → 映射 FAULT_TYPES 列表 → FAULT_CATEGORY_MAP

**FAULT_TYPES 列表**（31 种，index 0-30）定义在 `RCAgentEval/scripts/dashboard/backend/api/utils.py`。
**FAULT_CATEGORY_MAP**：5 类（PodChaos, StressChaos, HTTPFault, NetworkChaos, JVMChaos）。

```bash
cd /home/nn/SOTA-agents/RCAgentEval

# 填充 difficulty 元数据到 data 表
UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RCAgentEval/<db>.db \
python3 << 'PYEOF'
import json, sqlite3, networkx as nx
from pathlib import Path

FAULT_TYPES = [
    "PodKill", "PodFailure", "ContainerKill",
    "StressCPU", "StressMemory",
    "HTTPAbort", "HTTPDelay", "HTTPStatusCode",
    "NetworkDelay", "NetworkLoss", "NetworkDuplicate", "NetworkCorrupt", "NetworkPartition",
    "JVMStressThread", "JVMStressCPU", "JVMStressMemory",
    "JVMRuleDelay", "JVMRuleReturn", "JVMRuleException",
    "JVMCodeCacheFillingException", "JVMCpuCountException",
    "JVMDuplicateException", "JVMHumongousAllocationException",
    "JVMInflationException", "JVMSocketConnectException",
    "JVMSqliteException", "JVMThreadPoolException",
    "JVMRaisedException", "JVMMySQLDriverException",
    "JVMMySQLHikariException", "JVMMySQLException",
]

FAULT_CATEGORY_MAP = {
    "PodKill": "PodChaos", "PodFailure": "PodChaos", "ContainerKill": "PodChaos",
    "StressCPU": "StressChaos", "StressMemory": "StressChaos",
    "HTTPAbort": "HTTPFault", "HTTPDelay": "HTTPFault", "HTTPStatusCode": "HTTPFault",
    "NetworkDelay": "NetworkChaos", "NetworkLoss": "NetworkChaos",
    "NetworkDuplicate": "NetworkChaos", "NetworkCorrupt": "NetworkChaos", "NetworkPartition": "NetworkChaos",
}
for ft in FAULT_TYPES:
    if ft.startswith("JVM"):
        FAULT_CATEGORY_MAP.setdefault(ft, "JVMChaos")

conn = sqlite3.connect('<db>.db')  # 替换为实际 DB 路径
rows = conn.execute("SELECT id, source, meta FROM data").fetchall()

for row_id, source, meta_str in rows:
    meta = json.loads(meta_str or '{}')
    data_dir = meta.get('source_data_dir', '')
    if not data_dir:
        continue

    difficulty = {}
    # 从 causal_graph.json 计算 SPL, n_svc, n_edge
    cg_path = Path(data_dir) / "causal_graph.json"
    if cg_path.exists():
        cg = json.load(open(cg_path))
        nodes = cg.get("nodes", [])
        edges = cg.get("edges", [])
        c2s = cg.get("component_to_service", {})
        services = set(c2s.values()) | set(n.get("component", "").split("|")[-1] for n in nodes)
        difficulty["n_svc"] = len(services)
        difficulty["n_edge"] = len(edges)
        # SPL via networkx
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e["source"], e["target"])
        rcs = [r.get("component", "") for r in cg.get("root_causes", [])]
        alarms = [a.get("component", "") for a in cg.get("alarm_nodes", [])]
        min_spl = float("inf")
        for rc in rcs:
            for alarm in alarms:
                if G.has_node(rc) and G.has_node(alarm):
                    try:
                        spl = nx.shortest_path_length(G, rc, alarm)
                        min_spl = min(min_spl, spl)
                    except nx.NetworkXNoPath:
                        pass
        difficulty["spl"] = min_spl if min_spl != float("inf") else 0

    # 从 injection.json 获取 fault_type
    inj_path = Path(data_dir) / "injection.json"
    if inj_path.exists():
        inj = json.load(open(inj_path))
        ft_idx = inj.get("ground_truth", {}).get("fault_type")
        if ft_idx is not None and 0 <= ft_idx < len(FAULT_TYPES):
            ft = FAULT_TYPES[ft_idx]
            difficulty["fault_type"] = ft
            difficulty["fault_category"] = FAULT_CATEGORY_MAP.get(ft, "Unknown")
        rc_services = inj.get("ground_truth", {}).get("service", [])
        difficulty["root_cause_service"] = ",".join(rc_services)

    meta["difficulty"] = difficulty
    conn.execute("UPDATE data SET meta=? WHERE id=?", (json.dumps(meta, ensure_ascii=False), row_id))

conn.commit()
conn.close()
print("Done: difficulty metadata populated")
PYEOF
```

---

### Step 8：Dashboard 可视化

```bash
cd /home/nn/SOTA-agents/RCAgentEval

# 确保 .env 中 UTU_DB_URL 是绝对路径
grep UTU_DB_URL .env
# 应该是：UTU_DB_URL=sqlite:////home/nn/SOTA-agents/RCAgentEval/<agentname>_init.db

# 启动 dashboard
nohup uv run python scripts/dashboard/run_dashboard.py --mode prod --port 8001 \
  > /tmp/dashboard.log 2>&1 &

# 在本地浏览器访问（需要 SSH 端口转发）
# 本地执行：ssh -L 8001:localhost:8001 nn@10.10.10.140
# 然后访问：http://localhost:8001
```

---

### Step 9：N-gram 和 Triplet Coherence 分析

评测完成后，可以通过 Dashboard API 或直接调用分析模块进行行为分析。

**N-gram 分析**（`utu/eval/analysis/ngram_analysis.py`）：
- 从 trajectory 提取 `data_type:intent` token 序列
- 计算词汇表、频率/准确率相关性、独占模式、首 token 分析、转移矩阵
- API: `GET /analysis/ngrams?exp_id=<exp_id>`

**Triplet Coherence 分析**（`utu/eval/analysis/triplet_coherence.py`）：
- Internal: T→A（service/datatype match）、A→R（effectiveness/GT discovery）
- External: T→T（progression）、R→T（utilization）、A→A（drift）
- 使用完整 GT（从 causal_graph.json 加载 service nodes/edges/root causes/alarm nodes）
- API: `GET /analysis/coherence?exp_id=<exp_id>`
- 已知限制：T→T advancing=0%，因为 think_tool reflections 不含 ts-* 服务名

**通过代码直接分析**：
```python
from utu.eval.analysis.extractor import extract_steps, GTContext
from utu.eval.analysis.ngram_analysis import analyze_single_sample, analyze_ngrams
from utu.eval.analysis.triplet_coherence import analyze_coherence

# 加载样本 trajectory 和 GT，然后调用分析函数
steps = extract_steps(trajectories, gt_context=gt_ctx)
ngram_result = analyze_single_sample(steps, correct=sample.correct)
coherence = analyze_coherence(all_step_lists)
```

---

## 三、已知坑与解决方案

### 坑 1：data_dir 路径错误
**现象**：agent 找不到 parquet 文件，工具调用返回空结果或报错。
**原因**：agent 配置中 `data_dir` 指向根数据目录，而不是具体的 incident 子目录。
**解决**：`run_rollout.py` 从 `augmented_question` 用正则提取正确路径：
```python
m = re.search(r"stored in[:\s]+`([^`]+)`", s.augmented_question or "")
data_dir = m.group(1) if m else ""
```

---

### 坑 2：response 被 markdown 包裹
**现象**：judge 评分全 0，`parse_causal_graph` 返回 None。
**原因**：LLM 输出了 ` ```json {...} ``` `，`json.loads()` 直接解析失败。
**解决**：在 `agent_runner.py` 输出前剥离 markdown：
```python
def strip_markdown_json(text: str) -> str:
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

result = {"output": strip_markdown_json(compressed_research), ...}
```
**修复已有 DB 数据**：
```python
import re
m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', sample.response, re.DOTALL)
if m:
    sample.response = m.group(1).strip()
```

---

### 坑 3：run_eval_judge.py 创建空白新记录
**现象**：judge 运行后产生新的 id（如 11-20），这些记录 response 为空，所有评分为 0。
**原因**：`BaseBenchmark.__init__` 调用 `dataset.load()`，按 config 的 `agent_type`/`model_name` 过滤，不匹配时从 DatasetSample 创建新的空白 EvaluationSample。
**解决**：**不要用 `run_eval_judge.py`**，改用 `scripts/rejudge_samples.py`（直接操作已有记录）。
**如何清理脏记录**：
```python
from sqlmodel import Session, select, create_engine
from utu.eval.data import EvaluationSample
engine = create_engine('sqlite:///...')
with Session(engine) as s:
    judged = s.exec(select(EvaluationSample).where(EvaluationSample.stage == 'judged')).all()
    for j in judged:
        if not j.response:   # 空 response 说明是脏记录
            s.delete(j)
    s.commit()
```

---

### 坑 4：DB 路径使用相对路径
**现象**：dashboard 空白，API 返回空列表；或不同目录运行时读到空 DB。
**原因**：`sqlite:///filename.db` 是相对路径，不同工作目录下解析到不同文件。
**解决**：所有地方统一使用 4 个斜杠的绝对路径：
```
sqlite:////home/nn/SOTA-agents/RCAgentEval/<name>.db
```

---

### 坑 5：rejudge 共享 DB 时重复 judge
**现象**：rejudge 处理了所有 agent 的数据，而非只处理目标 agent。
**原因**：`rejudge_samples.py` 不指定 `--exp_id` 时查询所有 `stage="rollout"+"judged"` 的样本。
**解决**：共享 DB 时必须指定 `--exp_id`：
```bash
uv run python scripts/rejudge_samples.py --exp_id thinkdepthai-kimi-k2
```
**注意**：rejudge 现在同时查询 `stage="rollout"` 和 `stage="judged"`，无需手动改 stage。

---

### 坑 6：Dashboard 分布图全显示 "unknown"
**现象**：Dashboard 的 Distribution by Fault Category 等图表全部显示 "unknown"。
**原因**：`data` 表的 `meta.difficulty` 未填充，`get_difficulty_from_meta()` 返回默认值。
**解决**：运行 Step 7.5 的 difficulty 填充脚本。

---

### 坑 7：graph_metrics 路径错误
**现象**：读取 `meta.graph_metrics.node_f1` 返回 None。
**原因**：实际存储路径是 `meta.graph_metrics.primary.node_f1`（多一层 `primary`）。
**解决**：使用正确路径 `meta["graph_metrics"]["primary"]["node_f1"]`。

---

### 坑 8：cp DB 覆盖已有修改（仅 SQLite 旧模式）
**现象**：在 RCAgentEval 填充了 difficulty，然后从 RolloutRunner cp DB 过来，difficulty 丢失。
**原因**：cp 是全量覆盖，RolloutRunner 的 DB 没有 difficulty 数据。
**解决**：使用 PostgreSQL 共享 DB 模式后此问题不再存在（所有组件连同一个数据库，无需 cp）。
SQLite 旧模式需严格按顺序操作：先 cp DB → 再填充 difficulty → 再 rejudge。

---

### 坑 9：API 余额不足（AccountOverdueError）
> 原坑 5
**现象**：部分样本在中途失败（exit_code=1），错误为 `AccountOverdueError` 或 `insufficient_quota`。
**解决**：充值后直接重跑，脚本自动只处理 `stage="init"` 的未完成样本。

---

### 坑 10：Doubao 模型名称
**正确格式**：`doubao-seed-2-0-pro-260215`（不是 `Doubao-Seed-2.0-pro`）。
**查询可用模型**：
```bash
curl https://ark.cn-beijing.volces.com/api/v3/models \
  -H "Authorization: Bearer <API_KEY>"
```

---

## 四、新 Agent 接入 Checklist

接入新 agent 前，确认以下事项：

- [ ] agent 项目有独立 Python 环境（uv/conda），可以 `uv run python agent_runner.py`
- [ ] 阅读 agent 源码，了解其图拓扑/推理框架（LangGraph/CrewAI/AutoGen 等）
- [ ] 确定 prompt 注入点（system message? human message? config?）
- [ ] 确定工具替换方案（用 `rca_tools.py` 替换原有搜索工具）
- [ ] 编写 `agent_runner.py`：
  - [ ] 读取 stdin JSON（6 字段）
  - [ ] 替换 prompt 和工具
  - [ ] 输出纯 JSON（无 markdown 包裹）
  - [ ] trajectory 转换为 OpenAI role 格式
  - [ ] 接入 UsageTracker 采集 token 用量（见改进 4）
- [ ] 手动测试：`echo '<payload>' | uv run python agent_runner.py`
- [ ] 在 `configs/agents/<agentname>.yaml` 添加配置
- [ ] 单条冒烟测试：`run_rollout.py --limit 1`
- [ ] 验证 DB 写入：stage=rollout，response 为合法 JSON
- [ ] 全量跑完后走 judge → dashboard 流程

---

## 五、文件路径速查

| 文件 | 路径 |
|------|------|
| RolloutRunner 入口 | `RolloutRunner/scripts/run_rollout.py` |
| Agent 配置目录 | `RolloutRunner/configs/agents/` |
| RCA Prompts | `RolloutRunner/configs/prompts/rca.yaml` |
| thinkdepthai runner | `Deep_Research/agent_runner.py` |
| aiq runner | `aiq/agent_runner.py` |
| aiq 设计文档 | `RolloutRunner/docs/plans/2026-03-10-aiq-eval-design.md` |
| o3_deep_research runner | `test/openai_test/agent_runner.py` |
| o3_deep_research 设计文档 | `RolloutRunner/docs/plans/2026-03-10-o3-deep-research-eval-design.md` |
| RCAgentEval preprocess | `RCAgentEval/scripts/preprocess_only.py` |
| RCAgentEval rejudge | `RCAgentEval/scripts/rejudge_samples.py` |
| Dashboard 启动 | `RCAgentEval/scripts/dashboard/run_dashboard.py` |
| 数据集 symlink | `RCAgentEval/eval-data/<exp_id>/data_XXXXXXXX/` |
| UsageTracker | `RolloutRunner/src/usage_tracker.py` |
| Cost Metrics 计算 | `RolloutRunner/src/cost_metrics.py` |
| Backfill 脚本 | `RolloutRunner/scripts/backfill_cost_metrics.py` |

---

## 六、改进记录

### 改进 1：逐个写入 DB + 断点恢复（2026-03-05）

**问题**：原始 `run_batch` 使用 `asyncio.gather` 收集全部结果后才 `write_batch`，中断后已完成的样本全部丢失。

**解决**：
- `runner.py` 的 `run_batch` 新增 `on_complete` 回调参数
- `run_rollout.py` 在回调中调用 `write_result` 逐个写入 DB
- `load_samples` 只读 `stage='init'`，天然支持断点恢复

**代码变更**：`src/runner.py`、`scripts/run_rollout.py`

### 改进 2：fse26 数据集选择（2026-03-05）

**问题**：随机选择数据集可能覆盖面不够。

**解决**：从 `test.db` 的 `data` 表按 `dataset='fse26'` 筛选（1180 条中 1178 条有 causal_graph.json），随机抽样 100 条生成评测集。

### 改进 3：模型切换流程（2026-03-05）

切换 thinkdepthai 从 Doubao 到 Kimi K2：
1. 注释掉 `Deep_Research/.env` 中 Doubao 配置，添加 Kimi K2 配置
2. 修改 `Deep_Research/agent_runner.py` 中 `make_llm_call` 和 `make_compress_research` 的模型名
3. 修改 `Deep_Research/src/research_agent.py` 中模型名（注释旧的，添加新的）
4. 修改 `RolloutRunner/configs/agents/thinkdepthai.yaml` 的 `model_name`

**注意**：不要直接删除旧配置，而是注释掉，方便切换回去。

### 改进 4：开销统计（Cost Metrics）（2026-03-11）

**需求**：跨 agent 横向对比推理开销，两个核心指标：
1. **Total Token Count** — 推理总量（不区分 input/output/reasoning），衡量资源消耗
2. **Effective Rounds** — LLM→tool 交互轮次，衡量交互复杂度

**实现方案**：

1. **UsageTracker（`src/usage_tracker.py`）**：monkey-patch OpenAI SDK 的 `Completions.create`（sync+async），拦截所有 LLM 调用的 `response.usage`，线程安全累加。框架无关（LangChain/litellm/MetaChain/TaskWeaver/直接 OpenAI 都走这层）。

2. **cost_metrics 模块（`src/cost_metrics.py`）**：
   - `compute_effective_rounds(trajectory)` — 从 trajectory 统计带 tool_calls 的 assistant 消息数
   - `estimate_token_count(trajectory)` — 从字符数估算 token 数（~3 chars/token，fallback 用）
   - `build_cost_metrics(trajectory, usage, model, time_cost)` — 构建完整 cost_metrics dict

3. **agent_runner.py 接入**：每个 agent 顶部添加 4 行代码：
   ```python
   sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
   from src.usage_tracker import UsageTracker
   _tracker = UsageTracker()
   _tracker.install_openai_hooks()
   ```
   输出时添加 `"usage": _tracker.get_usage()`。

4. **数据流**：agent stdout `usage` → `runner.py` AgentResult → `db_writer.py` build_cost_metrics → `meta.cost_metrics`

5. **已接入 8 个 agent**：thinkdepthai, deerflowV2.0, Auto-Deep-Research, DeepResearchAgent, aiq, TaskWeaver, OpenRCA, o3-deepResearch

6. **回填旧数据**：`scripts/backfill_cost_metrics.py`（从 trajectory 估算，支持 --dry-run / --force）

**存储位置**：`evaluation_data.meta.cost_metrics`
```json
{
  "effective_rounds": 15,
  "total_tokens": 45000,
  "token_source": "actual",
  "usage": {"total_tokens": 45000, "prompt_tokens": 30000, "completion_tokens": 15000, "llm_call_count": 16},
  "rounds_detail": {"effective_rounds": 15, "total_assistant_messages": 16, "total_tool_calls": 45, "total_tool_results": 45, "total_messages": 107},
  "time_cost": 180.5,
  "model": "kimi-k2-0905-preview"
}
```

**代码变更**：`src/usage_tracker.py`（新增）、`src/cost_metrics.py`（新增）、`src/runner.py`、`src/db_writer.py`、`scripts/backfill_cost_metrics.py`（新增）、8 个 agent_runner.py

### 改进 5：空 Output 验证 + AIMD 自适应并发 + 串行执行（2026-03-14）

**问题 1**：Agent exit 0 但输出空 JSON (`{"output":"","trajectory":[]}`) → DB 写入 stage="rollout" → 样本永久锁死无法重跑。
**修复**：`runner.py` 在解析 JSON 后验证 output 和 trajectory 均非空，否则返回 None，DB 不写入，stage 保持 "init"。

**问题 2**：固定并发数无法适应 API 限速，速率超限时大量失败且无退避。
**修复**：替换 `asyncio.Semaphore` 为 `AdaptiveConcurrency`（AIMD）：capacity=1 慢启动，每 10 次连续成功 +1（上限 max_capacity），失败后 capacity=max(1, capacity//2) + 指数退避（5s→120s）。

**问题 3**：8 个 agent 同时并发，单 shubiaobiao API key 过载。
**修复**：新增 `run_eval_sequential.sh`（串行 bash 脚本）和 `launch_eval.sh`（tmux 启动 + 监控 window）。

**代码变更**：`src/runner.py`（验证 + AdaptiveConcurrency 类）、`configs/agents/*.yaml`（concurrency: 15→5）、`run_eval_sequential.sh`（新增）、`launch_eval.sh`（新增）

---

## 七、已完成/已接入的 Agent 测评

### 已接入 Agent 列表

| Agent | 架构 | 配置文件 | runner | 设计文档 |
|-------|------|---------|--------|---------|
| thinkdepthai | 单循环 LLM 驱动（llm_call ←→ tool_node → compress） | `agents/thinkdepthai.yaml` | `Deep_Research/agent_runner.py` | `2026-03-04-thinkdepthai-eval-design.md` |
| aiq | AIRA 5 节点流水线（generate → research → summarize → reflect → finalize） | `agents/aiq.yaml` | `aiq/agent_runner.py` | `2026-03-10-aiq-eval-design.md` |
| o3_deep_research | 单循环 while 循环（纯 OpenAI SDK，无 LangChain，3 tools 无 think_tool） | `agents/o3_deep_research.yaml` | `test/openai_test/agent_runner.py` | `2026-03-10-o3-deep-research-eval-design.md` |

### 已完成的测评结果

| Agent | 模型 | 样本数 | 数据集 | DB 文件 | exp_id |
|-------|------|--------|--------|---------|--------|
| thinkdepthai | doubao-seed-2-0-pro-260215 | 10 | 随机 | thinkdepthai_init.db | rollout_thinkdepthai |
| thinkdepthai | kimi-k2-0905-preview | 100 | fse26 | thinkdepthai-kimi-k2.db | rollout_thinkdepthai |

### thinkdepthai + kimi-k2 最终评测结果（94/100 完成）

**核心指标：**
- AC@1: 43.6% (41/94)
- Node F1: 0.617, Edge F1: 0.395, RC F1: 0.289, Path Reachability: 0.396

**按故障类别：**
| 类别 | 准确率 | 样本数 |
|------|--------|--------|
| HTTPFault | 49.1% | 57 |
| NetworkChaos | 57.1% | 7 |
| StressChaos | 50.0% | 2 |
| JVMChaos | 13.3% | 15 |
| PodChaos | 0% | 13 |

**运行统计：**
- 平均推理轮次：~23（范围 1-41）
- 平均耗时：~6 min/case（concurrency=2）
- 工具调用成功率：~97%
- 6 个样本失败（超时/API 错误）
- 主要工具：query_parquet_files（~18次/case）、think_tool（~3次/case）

---

## 八、快速分析命令

```bash
# AC@1 准确率
python3 -c "
import sqlite3, json
conn = sqlite3.connect('<db>.db')
rows = conn.execute('SELECT id, response, correct_answer FROM evaluation_data WHERE stage=\"rollout\"').fetchall()
correct = 0
for rid, resp, ans in rows:
    data = json.loads(resp)
    edges = data.get('edges', [])
    sources = set(e.get('source','') for e in edges)
    targets = set(e.get('target','') for e in edges)
    roots = sources - targets or set(n.get('component','') for n in data.get('nodes',[]))
    if roots & set(a.strip() for a in ans.split(',')): correct += 1
print(f'AC@1: {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%')
"

# 工具调用精确统计
python3 << 'EOF'
import sqlite3, json
from collections import Counter
conn = sqlite3.connect('<db>.db')
rows = conn.execute('SELECT id, trajectories FROM evaluation_data WHERE stage="rollout"').fetchall()
categories = Counter()
for rid, traj_str in rows:
    traj = json.loads(traj_str or '[]')
    call_id_to_name = {}
    for m in traj:
        if m.get('role') == 'assistant':
            for tc in m.get('tool_calls', []):
                call_id_to_name[tc.get('id', '')] = tc.get('function', {}).get('name', 'unknown')
    for m in traj:
        if m.get('role') == 'tool':
            name = call_id_to_name.get(m.get('tool_call_id', ''), 'unknown')
            content = m.get('content', '')
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and 'error' in parsed:
                    categories[f'{name}: FAIL'] += 1
                else:
                    categories[f'{name}: OK'] += 1
            except:
                categories[f'{name}: OK'] += 1
for cat, cnt in categories.most_common():
    print(f'  {cnt:>4}  {cat}')
EOF
```
