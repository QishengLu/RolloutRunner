#!/usr/bin/env python
"""Run middleware v3 experiment on ALL thinkdepthai Qwen failure cases (105).

Serial execution with 429 retry (30-min backoff). Designed to run in tmux.

Usage:
    cd /home/nn/SOTA-agents/RolloutRunner
    MW_OPENAI_API_KEY=sk-xxx uv run python scripts/run_mw_v3_experiment.py
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.runner import AgentResult, run_agent
from src.db_writer import write_result
from src.cost_metrics import build_cost_metrics
from src.dataset import get_engine

from sqlalchemy import text
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

EXP_ID = "thinkdepthai-qwen3.5-plus-mw-v3"
SOURCE_EXP_ID = "thinkdepthai-qwen3.5-plus"
AGENT_TYPE = "thinkdepthai"
MODEL_NAME = "qwen3.5-plus"
CMD_BASE = ["/home/nn/SOTA-agents/Deep_Research/.venv/bin/python", "-u", "agent_runner.py"]
CWD = "/home/nn/SOTA-agents/Deep_Research"
TIMEOUT = None  # No timeout — let agent run to natural completion
DATA_DIR = "/home/nn/SOTA-agents/RolloutRunner/data"
LOG_DIR = "/home/nn/SOTA-agents/RolloutRunner/logs/mw-v3"

FAILURE_INDEXES = [
    33, 99, 130, 156, 247, 281, 283, 315, 323, 339, 341, 572, 579,
    755, 784, 804, 807, 832, 860, 864, 1114, 1140, 1143, 1195, 1218,
    1254, 1371, 1394, 1421, 1435, 1459, 1495, 1515, 1814, 1846, 1880,
    1917, 1934, 1948, 2092, 2130, 2211, 2231, 2253, 2258, 2285, 2390,
    2512, 2598, 2641, 2678, 2682, 2700, 2713, 2715, 2716, 2836, 2988,
    3059, 3112, 3114, 3120, 3125, 3128, 3138, 3219, 3222, 3278, 3393,
    3524, 3552, 3592, 3605, 3622, 3673, 3716, 3760, 3776, 3868, 3878,
    3920, 3955, 4032, 4055, 4070, 4081, 4151, 4229, 4258, 4309, 4311,
    4353, 4363, 4375, 4463, 4510, 4519, 4530, 4617, 4707, 4732, 4758,
    4789, 4841, 4893,
]

RETRY_WAIT = 1800  # 30 minutes on 429
MAX_RETRIES = 999  # effectively unlimited — keep retrying for hours
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── Prompts (same as run_rollout.py) ───────────────────────────────────────

def load_rca_prompts() -> dict:
    import yaml
    prompt_path = Path(__file__).parent.parent / "configs" / "prompts" / "rca.yaml"
    with open(prompt_path) as f:
        return yaml.safe_load(f)


def build_payload(augmented_question: str, prompts: dict, data_dir: str) -> dict:
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    # Extract incident description for user prompt
    incident_desc = augmented_question

    return {
        "question": augmented_question,
        "system_prompt": prompts["RCA_ANALYSIS_SP"].format(date=today),
        "user_prompt": prompts["RCA_ANALYSIS_UP"].format(
            incident_description=incident_desc
        ),
        "compress_system_prompt": prompts.get("COMPRESS_FINDINGS_SP", ""),
        "compress_user_prompt": prompts.get("COMPRESS_FINDINGS_UP", ""),
        "data_dir": data_dir,
    }


# ── Environment ────────────────────────────────────────────────────────────

def build_env() -> dict[str, str]:
    """Build clean subprocess environment with middleware enabled."""
    skip = {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_EXE"}
    env = {k: v for k, v in os.environ.items() if k not in skip}

    if "PATH" in env:
        paths = env["PATH"].split(":")
        paths = [p for p in paths if "/miniconda3/envs/" not in p
                 and "RolloutRunner/.venv" not in p]
        env["PATH"] = ":".join(paths)

    # API key from command line (never hardcode)
    api_key = os.environ.get("MW_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("MW_OPENAI_API_KEY must be set (pass via command line)")
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = OPENAI_BASE_URL

    # Middleware v2 settings
    env["ENABLE_MIDDLEWARE"] = "1"
    env["MIDDLEWARE_DEFICIENCIES"] = "B1,B2,B3,B5,M1,M2,M3,M4"
    env["MW_MAX_INTERVENTIONS"] = "2"
    env["MW_CHECK_POINTS"] = "37,44"
    env["MW_MAX_PER_DIM"] = "1"
    env["MW_MAX_CONCLUSION"] = "1"
    env["RCA_MODEL"] = MODEL_NAME

    return env


# ── DB helpers ─────────────────────────────────────────────────────────────

def get_source_samples() -> list[dict]:
    """Get failure case samples from source experiment."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, dataset_index, augmented_question, meta,
                   dataset, source, raw_question, level, correct_answer, file_name
            FROM evaluation_data
            WHERE exp_id = :exp_id AND stage = 'judged' AND correct = false
              AND dataset_index = ANY(:indexes)
            ORDER BY dataset_index
        """), {
            "exp_id": SOURCE_EXP_ID,
            "indexes": FAILURE_INDEXES,
        }).fetchall()

    samples = []
    for r in rows:
        samples.append({
            "source_id": r[0],
            "dataset_index": r[1],
            "augmented_question": r[2],
            "meta": r[3],
            "dataset": r[4],
            "source": r[5],
            "raw_question": r[6],
            "level": r[7],
            "correct_answer": r[8],
            "file_name": r[9],
        })
    return samples


def ensure_init_samples(samples: list[dict]):
    """Create init samples in DB for MW v2 experiment if not exist."""
    engine = get_engine()
    with engine.connect() as conn:
        existing = conn.execute(text("""
            SELECT dataset_index FROM evaluation_data
            WHERE exp_id = :exp_id
        """), {"exp_id": EXP_ID}).fetchall()
        existing_indexes = {r[0] for r in existing}

        for s in samples:
            if s["dataset_index"] in existing_indexes:
                continue
            meta = s["meta"]
            if isinstance(meta, dict):
                meta = json.dumps(meta)
            conn.execute(text("""
                INSERT INTO evaluation_data
                    (exp_id, dataset_index, augmented_question, meta,
                     dataset, source, raw_question, level, correct_answer,
                     file_name, stage, agent_type, model_name)
                VALUES
                    (:exp_id, :idx, :aq, :meta,
                     :dataset, :source, :raw_question, :level, :correct_answer,
                     :file_name, 'init', :agent_type, :model_name)
            """), {
                "exp_id": EXP_ID,
                "idx": s["dataset_index"],
                "aq": s["augmented_question"],
                "meta": meta,
                "dataset": s["dataset"],
                "source": s["source"],
                "raw_question": s["raw_question"],
                "level": s["level"],
                "correct_answer": s["correct_answer"],
                "file_name": s["file_name"],
                "agent_type": AGENT_TYPE,
                "model_name": MODEL_NAME,
            })
        conn.commit()

    logger.info(f"Init samples ready: {len(samples)} total, "
                f"{len(samples) - len(existing_indexes)} newly created")


def get_pending_samples() -> list[dict]:
    """Get stage='init' samples for MW v2 experiment."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, dataset_index, augmented_question
            FROM evaluation_data
            WHERE exp_id = :exp_id AND stage = 'init'
            ORDER BY dataset_index
        """), {"exp_id": EXP_ID}).fetchall()

    return [{"id": r[0], "dataset_index": r[1], "augmented_question": r[2]}
            for r in rows]


# ── Main loop ──────────────────────────────────────────────────────────────

async def run_single(sample: dict, prompts: dict, env: dict) -> bool:
    """Run one sample. Returns True if succeeded, False if should retry."""
    idx = sample["dataset_index"]
    aq = sample["augmented_question"] or ""

    # Extract data_dir
    m = re.search(r"stored in[:\s]+`([^`]+)`", aq)
    data_dir = m.group(1) if m else ""
    if not data_dir:
        logger.warning(f"[idx={idx}] Cannot extract data_dir, skipping")
        return True  # skip, don't retry

    payload = build_payload(aq, prompts, data_dir)
    t0 = time.time()

    # Per-sample verbose log file
    log_file = os.path.join(LOG_DIR, f"idx_{idx}_sample_{sample['id']}.log")
    cmd = CMD_BASE + ["--log-file", log_file]

    result = await run_agent(
        sample_id=sample["id"],
        payload=payload,
        cmd=cmd,
        cwd=CWD,
        timeout=TIMEOUT,
        env=env,
    )

    if result is None:
        elapsed = time.time() - t0
        if elapsed < 60:
            # Fast crash — likely 429 or API error → retry after backoff
            logger.warning(f"[idx={idx}] Agent crashed after {elapsed:.0f}s, likely 429")
            return False
        else:
            # Long run then crash — agent internal error → skip, move on
            logger.warning(f"[idx={idx}] Agent crashed after {elapsed:.0f}s, skipping (internal error)")
            return True

    # Check for 429 — exact match on dashscope API error format
    traj_str = json.dumps(result.trajectory)
    combined = result.output + traj_str
    if "Error code: 429" in combined:
        logger.warning(f"[idx={idx}] Detected 429 rate limit, NOT writing to DB")
        return False  # signal retry

    # Validate output
    if not result.output or not result.trajectory:
        logger.warning(f"[idx={idx}] Empty output/trajectory, skipping write")
        return True  # don't retry empty results

    # Write to DB
    ok = write_result(
        result=result,
        exp_id=EXP_ID,
        agent_type=AGENT_TYPE,
        model_name=MODEL_NAME,
    )

    if ok:
        cm = build_cost_metrics(
            trajectory=result.trajectory,
            usage=result.usage if result.usage else None,
            model=MODEL_NAME,
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
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Middleware v3 Experiment — thinkdepthai × Qwen ALL failure cases")
    logger.info(f"exp_id: {EXP_ID}")
    logger.info(f"Failure cases: {len(FAILURE_INDEXES)}")
    logger.info(f"Per-sample logs: {LOG_DIR}")
    logger.info("=" * 60)

    prompts = load_rca_prompts()
    env = build_env()

    # Step 1: Get source samples and create init records
    source_samples = get_source_samples()
    logger.info(f"Found {len(source_samples)} failure cases in source experiment")
    ensure_init_samples(source_samples)

    # Step 2: Process pending samples with AIMD adaptive concurrency
    from src.runner import AdaptiveConcurrency
    CONCURRENCY = 12
    INITIAL_CAPACITY = 6

    for attempt in range(MAX_RETRIES + 1):
        pending = get_pending_samples()
        if not pending:
            logger.info("All samples completed!")
            break

        logger.info(f"Attempt {attempt + 1}: {len(pending)} samples pending")
        ac = AdaptiveConcurrency(max_capacity=CONCURRENCY, initial_capacity=INITIAL_CAPACITY)
        logger.info(f"[AIMD] Starting with capacity={ac.capacity}, max={CONCURRENCY}")

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
                ok = await run_single(sample, prompts, env)
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
            break

    # Final summary
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT stage, COUNT(*) FROM evaluation_data
            WHERE exp_id = :exp_id GROUP BY stage ORDER BY stage
        """), {"exp_id": EXP_ID}).fetchall()

    logger.info("=" * 60)
    logger.info("Final status:")
    for r in rows:
        logger.info(f"  {r[0]}: {r[1]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
