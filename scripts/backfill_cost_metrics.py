#!/usr/bin/env python
"""
backfill_cost_metrics.py — 从已有 trajectory 数据回填 cost_metrics

用法：
    UTU_DB_URL=sqlite:////abs/path/to/xxx.db python scripts/backfill_cost_metrics.py

功能：
    遍历 DB 中所有 stage="rollout" 或 stage="judged" 的样本，
    从 trajectories 列解析 trajectory，计算 cost_metrics（effective_rounds + estimated tokens），
    写入 meta.cost_metrics。

注意：
    - 对已有 meta.cost_metrics.usage（实际 token 数据）的样本，保留实际数据
    - 仅对缺少 cost_metrics 或需要更新的样本写入
    - 支持 --force 强制覆盖已有 cost_metrics
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from src.cost_metrics import build_cost_metrics
from src.dataset import EvaluationSample, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def backfill(force: bool = False, dry_run: bool = False):
    engine = get_engine()
    updated = 0
    skipped = 0
    errors = 0

    with Session(engine) as session:
        stmt = select(EvaluationSample).where(
            EvaluationSample.stage.in_(["rollout", "judged"])
        )
        samples = session.exec(stmt).all()
        logger.info(f"Found {len(samples)} samples to process")

        for sample in samples:
            try:
                # Parse trajectories
                traj_str = sample.trajectories or ""
                if not traj_str:
                    skipped += 1
                    continue

                trajectory = json.loads(traj_str)

                # Parse meta
                meta = sample.meta or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}

                # Skip if already has cost_metrics and not forcing
                existing = meta.get("cost_metrics")
                if existing and not force:
                    skipped += 1
                    continue

                # Preserve actual usage if present
                existing_usage = None
                if existing and existing.get("token_source") == "actual":
                    existing_usage = existing.get("usage")

                cost_metrics = build_cost_metrics(
                    trajectory=trajectory,
                    usage=existing_usage,
                    model=sample.model_name or "",
                    time_cost=sample.time_cost or 0.0,
                )

                meta["cost_metrics"] = cost_metrics

                if dry_run:
                    logger.info(
                        f"[DRY-RUN] id={sample.id} agent={sample.agent_type} "
                        f"rounds={cost_metrics['effective_rounds']} "
                        f"tokens={cost_metrics['total_tokens']} "
                        f"source={cost_metrics['token_source']}"
                    )
                else:
                    sample.meta = meta
                    flag_modified(sample, "meta")
                    session.add(sample)

                updated += 1

            except Exception as e:
                logger.error(f"Error processing sample id={sample.id}: {e}")
                errors += 1

        if not dry_run:
            session.commit()

    logger.info(
        f"Done: updated={updated}, skipped={skipped}, errors={errors}"
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill cost_metrics from trajectory data")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cost_metrics")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    backfill(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
