#!/bin/bash
# Monitor mw-v3 experiment progress every 30 minutes
# Run in tmux: tmux new-session -d -s mw-v3-monitor bash /path/to/monitor_mw_v3.sh

ABSTRACT="/home/nn/SOTA-agents/RolloutRunner/logs/mw-v3/abstract.md"
DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
EXP_ID="thinkdepthai-qwen3.5-plus-mw-v3"
LOG_DIR="/home/nn/SOTA-agents/RolloutRunner/logs/mw-v3"

while true; do
    sleep 1800  # 30 minutes

    # Query DB for progress
    STATS=$(cd /home/nn/SOTA-agents/RCAgentEval && uv run python -c "
import os
os.environ['UTU_DB_URL'] = '$DB_URL'
from sqlalchemy import create_engine, text
engine = create_engine(os.environ['UTU_DB_URL'])
with engine.connect() as conn:
    rows = conn.execute(text('''
        SELECT stage, COUNT(*) FROM evaluation_data
        WHERE exp_id = '$EXP_ID'
        GROUP BY stage ORDER BY stage
    ''')).fetchall()
    stages = {r[0]: r[1] for r in rows}
    init = stages.get('init', 0)
    rollout = stages.get('rollout', 0)
    total = init + rollout
    print(f'{rollout}|{total}|{init}')
" 2>/dev/null)

    DONE=$(echo "$STATS" | cut -d'|' -f1)
    TOTAL=$(echo "$STATS" | cut -d'|' -f2)
    REMAINING=$(echo "$STATS" | cut -d'|' -f3)

    # Count active log files (written in last 10 min = actively running)
    ACTIVE=$(find "$LOG_DIR" -name "*.log" -mmin -10 | wc -l)

    # Count total log files
    LOG_COUNT=$(find "$LOG_DIR" -name "*.log" | wc -l)

    NOW=$(date '+%Y-%m-%d %H:%M')

    # Append to abstract
    echo "| $NOW | $DONE | - | - | $REMAINING | logs=$LOG_COUNT, active=$ACTIVE |" >> "$ABSTRACT"

    echo "[$NOW] Done=$DONE, Remaining=$REMAINING, Active=$ACTIVE"

    # Exit if all done
    if [ "$REMAINING" = "0" ] 2>/dev/null; then
        echo "[$NOW] All samples completed!"
        echo "| $NOW | **ALL DONE** | | | 0 | |" >> "$ABSTRACT"
        break
    fi

    # Exit if no mw-v3 experiment process running
    if ! pgrep -f "run_mw_v3_experiment" > /dev/null 2>&1; then
        echo "[$NOW] Experiment process not running, stopping monitor."
        echo "| $NOW | **STOPPED** | | | $REMAINING | experiment process exited |" >> "$ABSTRACT"
        break
    fi
done
