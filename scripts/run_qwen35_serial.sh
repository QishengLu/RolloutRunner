#!/bin/bash
# Serial runner for openrca → aiq → taskweaver on qwen3.5-plus via Coding Plan.
# Each agent uses run_rollout_with_retry.py (AIMD slow-start, 429 retry with 30min backoff).
# Logs to logs/qwen3.5-serial/<agent>.log. Run inside tmux for persistence.

set -u

REPO=/home/nn/SOTA-agents/RolloutRunner
LOG_DIR="$REPO/logs/qwen3.5-serial"
mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
export UTU_DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY must be set before running" >&2
  exit 1
fi

cd "$REPO"

run_one() {
  local agent_cfg=$1
  local source_exp_id=$2
  local log="$LOG_DIR/${source_exp_id}.log"

  echo "$(date +'%F %T') === START $source_exp_id ===" | tee -a "$log"
  uv run python scripts/run_rollout_with_retry.py \
    --agent "$agent_cfg" \
    --source_exp_id "$source_exp_id" \
    --max_concurrency 3 \
    --initial_concurrency 3 \
    >> "$log" 2>&1
  echo "$(date +'%F %T') === END   $source_exp_id (exit=$?) ===" | tee -a "$log"
}

run_one openrca-qwen    openrca-qwen3.5-plus
run_one aiq             aiq-qwen3.5-plus
run_one taskweaver      taskweaver-qwen3.5-plus

echo "$(date +'%F %T') === ALL THREE AGENTS DONE ==="
