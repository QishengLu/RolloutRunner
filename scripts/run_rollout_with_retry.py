#!/usr/bin/env python
"""Run rollout for a single agent with 429-retry + AIMD + persistent log.

Structurally mirrors run_mw_v3_experiment.py but is generic over YAML agent
configs (no middleware). Designed to run in tmux serially across multiple
agents.

Usage:
    cd /home/nn/SOTA-agents/RolloutRunner
    OPENAI_API_KEY=sk-sp-xxx \
        uv run python scripts/run_rollout_with_retry.py \
        --agent openrca-qwen \
        --source_exp_id openrca-qwen3.5-plus

Features:
- AIMD adaptive concurrency (from runner.AdaptiveConcurrency)
- Exact match on "Error code: 429" → sleep 1800s and retry batch
- Load-then-filter pending samples every attempt (natural resume)
- Writes to DB via db_writer.write_result
- cost_metrics logged per sample
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cost_metrics import build_cost_metrics
from src.dataset import get_engine
from src.db_writer import write_result
from src.runner import AdaptiveConcurrency, run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "configs" / "prompts" / "rca.yaml"
AGENTS_DIR = Path(__file__).parent.parent / "configs" / "agents"

RETRY_WAIT = 1800  # 30 minutes on 429
MAX_RETRIES = 999


def load_agent_config(name: str) -> dict:
    with open(AGENTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_rca_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_payload(question: str, prompts: dict, data_dir: str) -> dict:
    today = date.today().isoformat()
    return {
        "question": question,
        "system_prompt": prompts["RCA_ANALYSIS_SP"].format(date=today),
        "user_prompt": prompts["RCA_ANALYSIS_UP"].format(incident_description=question),
        "compress_system_prompt": prompts["COMPRESS_FINDINGS_SP"].format(date=today),
        "compress_user_prompt": prompts["COMPRESS_FINDINGS_UP"].format(
            date=today, incident_description=question
        ),
        "data_dir": data_dir,
    }


def clean_env() -> dict[str, str]:
    skip = {"VIRTUAL_ENV", "UV_ACTIVE", "UV_RUN_RECURSION_DEPTH",
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_EXE"}
    env = {k: v for k, v in os.environ.items() if k not in skip}
    if "PATH" in env:
        paths = env["PATH"].split(":")
        paths = [p for p in paths if "/miniconda3/envs/" not in p
                 and "RolloutRunner/.venv" not in p]
        env["PATH"] = ":".join(paths)
    return env


def get_pending_samples(exp_id: str) -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, dataset_index, augmented_question
            FROM evaluation_data
            WHERE exp_id = :exp_id AND stage = 'init'
            ORDER BY dataset_index
        """), {"exp_id": exp_id}).fetchall()
    return [{"id": r[0], "dataset_index": r[1], "augmented_question": r[2]}
            for r in rows]


async def run_single(
    sample: dict,
    cfg: dict,
    prompts: dict,
    env: dict,
    per_sample_log_dir: str | None = None,
) -> bool:
    """Run one sample. Returns True if success (or non-retryable skip),
    False if 429 detected and caller should retry the batch later.

    If per_sample_log_dir is set and the agent binary supports --log-file,
    append `--log-file <dir>/idx_<N>_sample_<ID>.log` to cmd for this run.
    Each sample gets its own verbose log file for real-time tailing.
    """
    idx = sample["dataset_index"]
    aq = sample["augmented_question"] or ""

    m = re.search(r"stored in[:\s]+`([^`]+)`", aq)
    data_dir = m.group(1) if m else ""
    if not data_dir:
        logger.warning(f"[idx={idx}] Cannot extract data_dir, skipping")
        return True

    payload = build_payload(aq, prompts, data_dir)
    t0 = time.time()

    # Per-sample --log-file injection (only if per_sample_log_dir given)
    cmd = list(cfg["cmd"])
    if per_sample_log_dir:
        os.makedirs(per_sample_log_dir, exist_ok=True)
        log_file = os.path.join(
            per_sample_log_dir, f"idx_{idx}_sample_{sample['id']}.log"
        )
        cmd = cmd + ["--log-file", log_file]

    result = await run_agent(
        sample_id=sample["id"],
        payload=payload,
        cmd=cmd,
        cwd=cfg["cwd"],
        timeout=cfg["timeout"],
        env=env,
    )

    if result is None:
        elapsed = time.time() - t0
        if elapsed < 60:
            logger.warning(f"[idx={idx}] Agent crashed after {elapsed:.0f}s, likely 429")
            return False
        logger.warning(f"[idx={idx}] Agent crashed after {elapsed:.0f}s, skipping (internal error)")
        return True

    traj_str = json.dumps(result.trajectory)
    combined = (result.output or "") + traj_str
    if "Error code: 429" in combined:
        logger.warning(f"[idx={idx}] Detected 429 rate limit, NOT writing to DB")
        return False

    if not result.output or not result.trajectory:
        logger.warning(f"[idx={idx}] Empty output/trajectory, skipping write")
        return True

    ok = write_result(
        result=result,
        exp_id=cfg["exp_id"],
        agent_type=cfg["agent_type"],
        model_name=cfg["model_name"],
    )
    if ok:
        cm = build_cost_metrics(
            trajectory=result.trajectory,
            usage=result.usage if result.usage else None,
            model=cfg["model_name"],
            time_cost=result.time_cost,
        )
        logger.info(
            f"[idx={idx}] DONE in {result.time_cost:.0f}s — "
            f"tokens={cm.get('total_tokens', '?')} "
            f"rounds={cm.get('effective_rounds', '?')} "
            f"source={cm.get('token_source', '?')}"
        )
    else:
        logger.error(f"[idx={idx}] DB write failed")
    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="configs/agents/<name>.yaml")
    parser.add_argument("--source_exp_id", required=True,
                        help="exp_id of stage=init samples to consume")
    parser.add_argument("--max_concurrency", type=int, default=None,
                        help="override YAML concurrency (default: use YAML)")
    parser.add_argument("--initial_concurrency", type=int, default=None,
                        help="AIMD slow-start initial capacity")
    parser.add_argument("--log_dir", default=None,
                        help="Per-sample --log-file output directory (agent must support --log-file). "
                             "Each sample writes to idx_<N>_sample_<ID>.log inside this dir.")
    args = parser.parse_args()

    cfg = load_agent_config(args.agent)
    prompts = load_rca_prompts()
    env = {**clean_env(), "RCA_MODEL": cfg["model_name"]}

    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in environment")

    max_cap = args.max_concurrency or cfg["concurrency"]
    init_cap = args.initial_concurrency or max_cap

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Agent: {cfg['name']}  exp_id: {cfg['exp_id']}")
    logger.info(f"Source exp_id: {args.source_exp_id}")
    logger.info(f"Concurrency: initial={init_cap} max={max_cap}")
    logger.info(f"Timeout: {cfg['timeout']} (None = unlimited)")
    logger.info(f"API base: {env.get('OPENAI_BASE_URL', '(unset)')}")
    logger.info(f"Per-sample log dir: {args.log_dir or '(disabled)'}")
    logger.info("=" * 60)

    for attempt in range(MAX_RETRIES + 1):
        pending = get_pending_samples(args.source_exp_id)
        if not pending:
            logger.info("All samples completed!")
            break

        logger.info(f"Attempt {attempt + 1}: {len(pending)} samples pending")
        ac = AdaptiveConcurrency(max_capacity=max_cap, initial_capacity=init_cap)
        logger.info(f"[AIMD] Starting with capacity={ac.capacity}, max={max_cap}")

        hit_429 = False
        completed = 0

        async def run_with_aimd(sample, idx_in_batch):
            nonlocal hit_429, completed
            if hit_429:
                return
            await ac.acquire()
            try:
                if hit_429:
                    return
                idx = sample["dataset_index"]
                logger.info(f"[{idx_in_batch+1}/{len(pending)}] Running idx={idx}...")
                ok = await run_single(
                    sample, cfg, prompts, env,
                    per_sample_log_dir=args.log_dir,
                )
                success = ok is True
                await ac.backoff_on_failure(success)
                await ac.release(success)
                if not ok:
                    hit_429 = True
                    logger.warning(
                        f"[idx={idx}] 429 hit. Will retry "
                        f"in {RETRY_WAIT // 60} minutes."
                    )
                else:
                    completed += 1
            except Exception:
                await ac.backoff_on_failure(False)
                await ac.release(False)
                raise

        tasks = [run_with_aimd(s, i) for i, s in enumerate(pending)]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Attempt {attempt + 1} done: {completed} completed, 429={hit_429}")

        if hit_429 and attempt < MAX_RETRIES:
            logger.info(f"Sleeping {RETRY_WAIT // 60} minutes before retry...")
            await asyncio.sleep(RETRY_WAIT)
        elif not hit_429:
            # No 429 but some may have been skipped/crashed — exit loop
            break

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT stage, COUNT(*) FROM evaluation_data
            WHERE exp_id = :exp_id GROUP BY stage ORDER BY stage
        """), {"exp_id": args.source_exp_id}).fetchall()

    logger.info("=" * 60)
    logger.info(f"Final status for {args.source_exp_id}:")
    for r in rows:
        logger.info(f"  {r[0]}: {r[1]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
