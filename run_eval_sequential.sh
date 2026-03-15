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
