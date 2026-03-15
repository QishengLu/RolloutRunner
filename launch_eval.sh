#!/bin/bash
# launch_eval.sh — 在 tmux 里启动串行实验 + 监控 window
# 用法：bash launch_eval.sh
# 恢复：tmux attach -t eval

SESSION="eval"
ROLLOUT_DIR="/home/nn/SOTA-agents/RolloutRunner"
DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "runner"

# runner window：串行跑所有 agent
tmux send-keys -t "$SESSION:runner" \
    "bash $ROLLOUT_DIR/run_eval_sequential.sh" Enter

# monitor window：每 30s 刷新 DB 进度（用 docker exec，无需安装 psql 客户端）
PG_CONTAINER=$(docker ps --filter "expose=5432" -q | head -1)
tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" \
    "watch -n 30 \"docker exec $PG_CONTAINER psql -U postgres -d SOTA-Agents -c 'SELECT exp_id, stage, COUNT(*) as cnt FROM evaluation_data GROUP BY exp_id, stage ORDER BY exp_id, stage'\"" Enter

tmux select-window -t "$SESSION:runner"
tmux attach -t "$SESSION"
