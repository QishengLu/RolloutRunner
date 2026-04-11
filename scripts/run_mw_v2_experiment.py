#!/usr/bin/env python
"""Run middleware v2 experiment on thinkdepthai Qwen failure cases.

Serial execution with 429 retry (30-min backoff). Designed to run in tmux.

Usage:
    cd /home/nn/SOTA-agents/RolloutRunner
    uv run python scripts/run_mw_v2_experiment.py
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

EXP_ID = "thinkdepthai-qwen3.5-plus-mw-v2"
SOURCE_EXP_ID = "thinkdepthai-qwen3.5-plus"
AGENT_TYPE = "thinkdepthai"
MODEL_NAME = "qwen3.5-plus"
CMD = ["/home/nn/SOTA-agents/Deep_Research/.venv/bin/python", "agent_runner.py"]
CWD = "/home/nn/SOTA-agents/Deep_Research"
TIMEOUT = 1800
DATA_DIR = "/home/nn/SOTA-agents/RolloutRunner/data"

FAILURE_INDEXES = [
    33, 156, 247, 755, 804, 807, 1394, 1798, 1917, 2130, 2211,
    2231, 2390, 2682, 2700, 2988, 3114, 3120, 3138, 4375, 4893,
]

RETRY_WAIT = 1800  # 30 minutes on 429
MAX_RETRIES = 999  # effectively unlimited — keep retrying for hours
OPENAI_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"

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
    env["MW_MAX_INTERVENTIONS"] = "5"
    env["MW_CHECK_INTERVAL"] = "5"
    env["MW_MIN_QUERIES"] = "10"
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

    result = await run_agent(
        sample_id=sample["id"],
        payload=payload,
        cmd=CMD,
        cwd=CWD,
        timeout=TIMEOUT,
        env=env,
    )

    if result is None:
        # Check stderr/logs for 429 — run_agent returns None on crash
        # Re-run briefly to capture stderr for 429 detection
        logger.warning(f"[idx={idx}] Agent returned None, checking if 429...")
        # Assume 429 if agent crashes very quickly (< 30s) — likely API error
        return False  # signal retry to be safe

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
    logger.info("=" * 60)
    logger.info("Middleware v2 Experiment — thinkdepthai × Qwen failure cases")
    logger.info(f"exp_id: {EXP_ID}")
    logger.info(f"Failure cases: {len(FAILURE_INDEXES)}")
    logger.info("=" * 60)

    prompts = load_rca_prompts()
    env = build_env()

    # Step 1: Get source samples and create init records
    source_samples = get_source_samples()
    logger.info(f"Found {len(source_samples)} failure cases in source experiment")
    ensure_init_samples(source_samples)

    # Step 2: Process pending samples serially with 429 retry
    for attempt in range(MAX_RETRIES + 1):
        pending = get_pending_samples()
        if not pending:
            logger.info("All samples completed!")
            break

        logger.info(f"Attempt {attempt + 1}: {len(pending)} samples pending")

        hit_429 = False
        for i, sample in enumerate(pending):
            idx = sample["dataset_index"]
            logger.info(f"[{i+1}/{len(pending)}] Running idx={idx}...")

            ok = await run_single(sample, prompts, env)
            if not ok:
                hit_429 = True
                logger.warning(
                    f"[idx={idx}] 429 hit. Stopping batch, will retry "
                    f"in {RETRY_WAIT // 60} minutes."
                )
                break

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
