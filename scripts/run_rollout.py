#!/usr/bin/env python
"""RolloutRunner 入口脚本。"""
import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading

from src.cost_metrics import build_cost_metrics
from src.dataset import load_samples
from src.db_writer import write_batch, write_result
from src.runner import run_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "configs" / "prompts" / "rca.yaml"
AGENTS_DIR = Path(__file__).parent.parent / "configs" / "agents"


def load_agent_config(agent_name: str) -> dict:
    path = AGENTS_DIR / f"{agent_name}.yaml"
    with open(path) as f:
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


def _clean_env() -> dict[str, str]:
    """构造干净的子进程环境，去掉 conda/venv/uv 残留，避免污染 agent 的 Python 环境。"""
    skip = {"VIRTUAL_ENV", "UV_ACTIVE", "UV_RUN_RECURSION_DEPTH",
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV"}
    env = {k: v for k, v in os.environ.items() if k not in skip}
    # 从 PATH 中去掉 conda env 和 RolloutRunner .venv 的 bin 目录
    if "PATH" in env:
        paths = env["PATH"].split(":")
        paths = [p for p in paths if "/miniconda3/envs/" not in p
                 and "RolloutRunner/.venv" not in p]
        env["PATH"] = ":".join(paths)
    return env


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="configs/agents/<name>.yaml")
    parser.add_argument(
        "--source_exp_id",
        default="rcabench_evaluation",
        help="从哪个 exp_id 的 stage=init 样本读取",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制样本数（调试用）")
    parser.add_argument("--dataset-index", type=int, default=None,
                        help="指定 dataset_index，确保多 agent 冒烟测试使用同一个 case")
    args = parser.parse_args()

    cfg = load_agent_config(args.agent)
    prompts = load_rca_prompts()

    logger.info(f"Agent: {cfg['name']}  exp_id: {cfg['exp_id']}")
    logger.info(f"Reading from exp_id={args.source_exp_id}, stage=init")

    samples = load_samples(args.source_exp_id)
    if args.dataset_index is not None:
        samples = [s for s in samples if s.dataset_index == args.dataset_index]
        logger.info(f"Filtered to dataset_index={args.dataset_index}: {len(samples)} sample(s)")
    if args.limit:
        samples = samples[: args.limit]
    logger.info(f"Loaded {len(samples)} samples (stage=init, pending rollout)")

    if not samples:
        logger.warning("No pending samples found, exiting.")
        return

    tasks = []
    for s in samples:
        # 从 augmented_question 提取 data_dir（由 RCAgentEval preprocess 嵌入）
        m = re.search(r"stored in[:\s]+`([^`]+)`", s.augmented_question or "")
        data_dir = m.group(1) if m else ""
        if not data_dir:
            logger.warning(f"[sample {s.id}] 无法从 augmented_question 提取 data_dir，留空")
        tasks.append(
            {
                "id": s.id,
                "payload": build_payload(s.augmented_question, prompts, data_dir),
            }
        )

    # 逐个写入 DB 的回调（线程安全计数 + 用量累计）
    lock = threading.Lock()
    counters = {"success": 0, "failure": 0}
    cumulative_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost_usd": 0.0,
        "total_time": 0.0,
        "total_rounds": 0,
        "actual_count": 0,
        "estimated_count": 0,
    }

    def on_complete(result):
        if result is None:
            with lock:
                counters["failure"] += 1
            return
        ok = write_result(
            result=result,
            exp_id=cfg["exp_id"],
            agent_type=cfg["agent_type"],
            model_name=cfg["model_name"],
        )

        # 计算本次 cost_metrics 并累加
        cm = build_cost_metrics(
            trajectory=result.trajectory,
            usage=result.usage if result.usage else None,
            model=cfg["model_name"],
            time_cost=result.time_cost,
        )

        with lock:
            if ok:
                counters["success"] += 1
            else:
                counters["failure"] += 1
            total = counters["success"] + counters["failure"]

            # 累加用量
            cumulative_usage["total_tokens"] += cm.get("total_tokens", 0)
            cumulative_usage["total_time"] += result.time_cost
            cumulative_usage["total_rounds"] += cm.get("effective_rounds", 0)
            if cm.get("token_source") == "actual":
                cumulative_usage["actual_count"] += 1
                usage_data = cm.get("usage", {})
                cumulative_usage["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
                cumulative_usage["completion_tokens"] += usage_data.get("completion_tokens", 0)
            else:
                cumulative_usage["estimated_count"] += 1
            cost_info = cm.get("cost_usd", {})
            cumulative_usage["total_cost_usd"] += cost_info.get("total", 0.0)

        # 单条日志：包含本次 token/cost 信息
        token_src = cm.get("token_source", "?")
        tokens = cm.get("total_tokens", 0)
        rounds = cm.get("effective_rounds", 0)
        cost_usd = cost_info.get("total", 0.0) if cost_info else 0.0
        logger.info(
            f"Progress: {total}/{len(tasks)} "
            f"(ok={counters['success']}, fail={counters['failure']}) | "
            f"sample={result.sample_id} "
            f"tokens={tokens:,}({token_src}) rounds={rounds} "
            f"cost=${cost_usd:.4f} time={result.time_cost:.1f}s"
        )

    await run_batch(
        samples=tasks,
        cmd=cfg["cmd"],
        cwd=cfg["cwd"],
        timeout=cfg["timeout"],
        concurrency=cfg["concurrency"],
        env=_clean_env(),
        on_complete=on_complete,
    )

    # ── 累计用量汇总 ────────────────────────────────────────────────────────
    cu = cumulative_usage
    logger.info("=" * 70)
    logger.info(f"Agent: {cfg['name']}  Model: {cfg['model_name']}")
    logger.info(f"Results: success={counters['success']}, failure={counters['failure']}")
    logger.info(f"Total tokens: {cu['total_tokens']:,} "
                f"(actual={cu['actual_count']}, estimated={cu['estimated_count']})")
    if cu["actual_count"] > 0:
        logger.info(f"  Prompt tokens: {cu['prompt_tokens']:,}  "
                    f"Completion tokens: {cu['completion_tokens']:,}")
    logger.info(f"Total cost: ${cu['total_cost_usd']:.4f} USD")
    logger.info(f"Total time: {cu['total_time']:.1f}s  "
                f"Avg time: {cu['total_time'] / max(counters['success'], 1):.1f}s/sample")
    logger.info(f"Total rounds: {cu['total_rounds']}  "
                f"Avg rounds: {cu['total_rounds'] / max(counters['success'], 1):.1f}/sample")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
