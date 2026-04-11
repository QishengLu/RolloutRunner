#!/bin/bash
# smoke_retry_qwen.sh — 每 30 分钟重试 OpenRCA 和 mABC 的 qwen3.5-plus 冒烟测试
# 两个都成功后自动退出
# 用法：bash smoke_retry_qwen.sh

set -uo pipefail

export UTU_DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
export OPENAI_API_KEY="$1"  # 从命令行参数传入

ROLLOUT_DIR="/home/nn/SOTA-agents/RolloutRunner"
PG_CONTAINER=$(docker ps --filter "expose=5432" -q | head -1)
LOG_DIR="$ROLLOUT_DIR/logs"
mkdir -p "$LOG_DIR"

check_done() {
    # 检查指定 exp_id 是否已有 rollout/judged 样本
    local exp_id="$1"
    local count
    count=$(docker exec "$PG_CONTAINER" psql -U postgres -d SOTA-Agents -t -c \
        "SELECT COUNT(*) FROM evaluation_data WHERE exp_id='$exp_id' AND stage IN ('rollout','judged')" 2>/dev/null | tr -d ' ')
    [[ "$count" -gt 0 ]]
}

OPENRCA_DONE=false
MABC_DONE=false
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "========================================"
    echo "=== Attempt #$ATTEMPT at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # --- OpenRCA ---
    if $OPENRCA_DONE || check_done "openrca-qwen3.5-plus"; then
        OPENRCA_DONE=true
        echo "[OpenRCA] Already succeeded, skipping."
    else
        echo "[OpenRCA] Running smoke test..."
        cd "$ROLLOUT_DIR"
        uv run python scripts/run_rollout.py \
            --agent openrca-qwen \
            --source_exp_id openrca-qwen3.5-plus \
            --limit 1 \
            2>&1 | tee "$LOG_DIR/openrca-qwen-smoke.log"

        if check_done "openrca-qwen3.5-plus"; then
            OPENRCA_DONE=true
            echo "[OpenRCA] SUCCESS!"
        else
            echo "[OpenRCA] Failed (likely 429 rate limit), will retry."
        fi
    fi

    # --- mABC ---
    if $MABC_DONE || check_done "mabc-qwen3.5-plus"; then
        MABC_DONE=true
        echo "[mABC] Already succeeded, skipping."
    else
        echo "[mABC] Running smoke test..."
        cd "$ROLLOUT_DIR"
        uv run python scripts/run_rollout.py \
            --agent mabc-qwen \
            --source_exp_id mabc-qwen3.5-plus \
            --limit 1 \
            2>&1 | tee "$LOG_DIR/mabc-qwen-smoke.log"

        if check_done "mabc-qwen3.5-plus"; then
            MABC_DONE=true
            echo "[mABC] SUCCESS!"
        else
            echo "[mABC] Failed (likely 429 rate limit), will retry."
        fi
    fi

    # --- 两个都完成则退出 ---
    if $OPENRCA_DONE && $MABC_DONE; then
        echo ""
        echo "========================================"
        echo "=== BOTH SMOKE TESTS PASSED ==="
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo "========================================"
        exit 0
    fi

    echo ""
    echo "Sleeping 30 minutes until next retry ($(date -d '+30 minutes' '+%H:%M:%S'))..."
    sleep 1800
done
