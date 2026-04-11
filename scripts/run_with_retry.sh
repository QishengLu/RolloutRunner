#!/bin/bash
# 自动重试 rollout 脚本：429 限流后等待恢复再重试
# 用法: bash scripts/run_with_retry.sh <agent_name> <exp_id> [wait_minutes]
#
# 逻辑：
#   1. 跑一轮 rollout
#   2. 检查 DB 是否还有 init 样本
#   3. 有 → 等 wait_minutes 分钟后重试
#   4. 没有 → 结束

AGENT="${1:?Usage: $0 <agent_name> <exp_id> [wait_minutes]}"
EXP_ID="${2:?Usage: $0 <agent_name> <exp_id> [wait_minutes]}"
WAIT_MIN="${3:-30}"

export UTU_DB_URL="${UTU_DB_URL:-postgresql://postgres:postgres@localhost:5433/SOTA-Agents}"

cd /home/nn/SOTA-agents/RolloutRunner

while true; do
    # 检查剩余 init 数量
    REMAINING=$(uv run python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.environ['UTU_DB_URL'])
with engine.connect() as conn:
    r = conn.execute(text(\"SELECT COUNT(*) FROM evaluation_data WHERE exp_id='${EXP_ID}' AND stage='init'\")).fetchone()
    print(r[0])
" 2>/dev/null)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining init samples: ${REMAINING}"

    if [ "${REMAINING}" = "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All samples completed! Exiting."
        break
    fi

    # 跑一轮
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rollout for ${AGENT} (${EXP_ID})..."
    uv run python scripts/run_rollout.py --agent "${AGENT}" --source_exp_id "${EXP_ID}" 2>&1

    # 再次检查是否还有剩余
    REMAINING=$(uv run python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.environ['UTU_DB_URL'])
with engine.connect() as conn:
    r = conn.execute(text(\"SELECT COUNT(*) FROM evaluation_data WHERE exp_id='${EXP_ID}' AND stage='init'\")).fetchone()
    print(r[0])
" 2>/dev/null)

    if [ "${REMAINING}" = "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All samples completed! Exiting."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${REMAINING} samples remaining. Waiting ${WAIT_MIN} minutes for quota recovery..."
    sleep $((WAIT_MIN * 60))
done
