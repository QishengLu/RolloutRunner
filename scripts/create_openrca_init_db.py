#!/usr/bin/env python
"""
从 thinkdepthai-kimi-k2.db 复制样本，创建 openrca-kimi-k2.db（stage='init', exp_id='rollout_openrca'）。
只保留 data_dir 路径仍然有效的样本。

Usage:
    uv run python scripts/create_openrca_init_db.py [--limit N]
"""
import argparse
import os
import re
import sqlite3
import shutil
from pathlib import Path

SRC_DB = Path(__file__).parent.parent / "thinkdepthai-kimi-k2.db"
DST_DB = Path(__file__).parent.parent / "openrca-kimi-k2.db"
NEW_EXP_ID = "rollout_openrca"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只复制前 N 条（调试用）")
    args = parser.parse_args()

    if not SRC_DB.exists():
        raise FileNotFoundError(f"Source DB not found: {SRC_DB}")

    if DST_DB.exists():
        print(f"[warn] {DST_DB} already exists, will be overwritten.")
        DST_DB.unlink()

    # Copy schema
    shutil.copy(SRC_DB, DST_DB)
    print(f"Copied schema from {SRC_DB} → {DST_DB}")

    src_conn = sqlite3.connect(SRC_DB)
    dst_conn = sqlite3.connect(DST_DB)

    try:
        src_cur = src_conn.cursor()
        dst_cur = dst_conn.cursor()

        # Clear destination evaluation_data
        dst_cur.execute("DELETE FROM evaluation_data")

        # Read all samples from source
        src_cur.execute("""
            SELECT id, created_at, updated_at, dataset, dataset_index, source,
                   raw_question, level, augmented_question, correct_answer,
                   file_name, meta, tags
            FROM evaluation_data
            ORDER BY id
        """)
        rows = src_cur.fetchall()

        if args.limit:
            rows = rows[: args.limit]

        valid = 0
        skipped = 0
        for row in rows:
            (
                _id, created_at, updated_at, dataset, dataset_index, source,
                raw_question, level, augmented_question, correct_answer,
                file_name, meta, tags,
            ) = row

            # Validate data_dir still exists
            m = re.search(r"stored in[:\s]+`([^`]+)`", augmented_question or "")
            if not m:
                print(f"  [skip] id={_id}: cannot extract data_dir from question")
                skipped += 1
                continue
            data_dir = m.group(1).strip()
            if not os.path.isdir(data_dir):
                print(f"  [skip] id={_id}: data_dir missing: {data_dir}")
                skipped += 1
                continue

            dst_cur.execute(
                """
                INSERT INTO evaluation_data
                    (created_at, updated_at, dataset, dataset_index, source,
                     raw_question, level, augmented_question, correct_answer,
                     file_name, meta, tags,
                     exp_id, agent_type, model_name, stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, updated_at, dataset, dataset_index, source,
                    raw_question, level, augmented_question, correct_answer,
                    file_name, meta, tags,
                    NEW_EXP_ID, "openrca", "kimi-k2-0905-preview", "init",
                ),
            )
            valid += 1

        dst_conn.commit()
        print(f"\nDone: {valid} samples written (stage='init'), {skipped} skipped.")
        print(f"Output DB: {DST_DB}")
        print(f"\nNext steps:")
        print(f"  # Smoke test (1 sample):")
        print(f"  UTU_DB_URL=sqlite:////{DST_DB} python scripts/run_rollout.py --agent openrca --source_exp_id {NEW_EXP_ID} --limit 1")
        print(f"  # Full run:")
        print(f"  UTU_DB_URL=sqlite:////{DST_DB} nohup python -u scripts/run_rollout.py --agent openrca --source_exp_id {NEW_EXP_ID} > openrca.log 2>&1 &")

    finally:
        src_conn.close()
        dst_conn.close()


if __name__ == "__main__":
    main()
