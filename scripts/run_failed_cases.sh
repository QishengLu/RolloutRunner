#!/bin/bash
# Run middleware experiment on Qwen's 21 failed cases
# Usage: ENABLE_MIDDLEWARE=1 bash scripts/run_failed_cases.sh

FAILED_INDEXES=(33 156 247 755 804 807 1394 1798 1917 2130 2211 2231 2390 2682 2700 2988 3114 3120 3138 4375 4893)

echo "Running ${#FAILED_INDEXES[@]} failed cases with middleware..."
echo "ENABLE_MIDDLEWARE=${ENABLE_MIDDLEWARE:-0}"

for idx in "${FAILED_INDEXES[@]}"; do
    echo "=== Running dataset_index=$idx ==="
    uv run python scripts/run_rollout.py \
        --agent thinkdepthai-qwen-mw \
        --source_exp_id thinkdepthai-qwen3.5-plus-mw \
        --dataset-index "$idx"
    echo "=== Done idx=$idx ==="
    echo
done

echo "All ${#FAILED_INDEXES[@]} cases completed."
