#!/bin/bash
# run_resample.sh — thinkdepthai-qwen3.5-plus failed cases × 2 resamples.
# Expects shell env: OPENAI_API_KEY, OPENAI_BASE_URL, RCA_MODEL (do NOT hardcode keys here).

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" || -z "${RCA_MODEL:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY / OPENAI_BASE_URL / RCA_MODEL must be set in env" >&2
  exit 1
fi

echo "[$(date '+%F %T')] === Resample 1 starting ==="
uv run python scripts/run_rollout.py \
  --agent thinkdepthai-qwen \
  --source_exp_id thinkdepthai-qwen3.5-plus-resample-1 \
  --exp_id thinkdepthai-qwen3.5-plus-resample-1 \
  --log-dir logs/resample-1 \
  2>&1 | tee -a logs/resample-1/driver.log

echo "[$(date '+%F %T')] === Resample 2 starting ==="
uv run python scripts/run_rollout.py \
  --agent thinkdepthai-qwen \
  --source_exp_id thinkdepthai-qwen3.5-plus-resample-2 \
  --exp_id thinkdepthai-qwen3.5-plus-resample-2 \
  --log-dir logs/resample-2 \
  2>&1 | tee -a logs/resample-2/driver.log

echo "[$(date '+%F %T')] === All resamples complete ==="
