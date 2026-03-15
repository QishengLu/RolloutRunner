import os
import sys
from sqlalchemy import create_engine, text

# DB Connection URL
db_url = "postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
engine = create_engine(db_url)

try:
    with engine.connect() as conn:
        print("--- Init Stage ---")
        query_init = text("SELECT exp_id, stage, COUNT(*) as count FROM evaluation_data WHERE stage = 'init' GROUP BY exp_id, stage ORDER BY exp_id")
        result_init = conn.execute(query_init)
        for row in result_init:
            print(f"exp_id: {row[0]}, stage: {row[1]}, count: {row[2]}")
            
        print("\n--- Rollout/Judged Stage ---")
        query_done = text("SELECT exp_id, stage, COUNT(*) as count FROM evaluation_data WHERE stage IN ('rollout', 'judged') GROUP BY exp_id, stage ORDER BY exp_id")
        result_done = conn.execute(query_done)
        for row in result_done:
            print(f"exp_id: {row[0]}, stage: {row[1]}, count: {row[2]}")

except Exception as e:
    print(f"Error querying database: {e}")
