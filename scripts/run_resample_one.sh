#!/bin/bash
# run_resample_one.sh N — run resample-N (N=1 or 2) for thinkdepthai-qwen3.5-plus.
# Requires shell env: OPENAI_API_KEY, OPENAI_BASE_URL, RCA_MODEL.

set -euo pipefail

N="${1:?usage: run_resample_one.sh <1|2>}"
cd "$(dirname "$0")/.."

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" || -z "${RCA_MODEL:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY / OPENAI_BASE_URL / RCA_MODEL must be set" >&2
  exit 1
fi

EXPID="thinkdepthai-qwen3.5-plus-resample-${N}"
LOGDIR="logs/resample-${N}"
mkdir -p "$LOGDIR"

echo "[$(date '+%F %T')] Launching ${EXPID} (concurrency=16, parallel with other resample)"
uv run python scripts/run_rollout.py \
  --agent thinkdepthai-qwen \
  --source_exp_id "${EXPID}" \
  --exp_id "${EXPID}" \
  --log-dir "${LOGDIR}" \
  2>&1 | tee -a "${LOGDIR}/driver.log"

echo "[$(date '+%F %T')] ${EXPID} done"
