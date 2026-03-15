import argparse
from sqlalchemy import create_engine, text
import json
import sys

# DB Connection URL
db_url = "postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
engine = create_engine(db_url)

def analyze(exp_id):
    try:
        with engine.connect() as conn:
            # Get the latest modified rollout
            query = text("""
                SELECT trajectories, meta, created_at, updated_at
                FROM evaluation_data 
                WHERE exp_id = :exp_id AND stage IN ('rollout', 'judged')
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            # result = conn.execute(query, {"exp_id": exp_id}).fetchone()
            # Try to fetch as dict to be sure
            result_proxy = conn.execute(query, {"exp_id": exp_id})
            result = result_proxy.fetchone() # This returns a Row object
            
            if not result:
                print(f"No rollout found for exp_id: {exp_id}")
                return

            trajectory = result[0]
            meta = result[1]
            
            print(f"--- Analysis for {exp_id} ---")
            
            # Access by index since it's a tuple-like or Row
            meta = result[1]
            trajectory = result[0]
            
            print(f"Trajectory Type: {type(trajectory)}")
            
            if isinstance(trajectory, str):
                try:
                    trajectory = json.loads(trajectory)
                except json.JSONDecodeError:
                    print("Trajectory is not valid JSON")
                    print(trajectory[:500])
                    # return

            # Cost Metrics
            cost_metrics = meta.get('cost_metrics', {})
            print("Cost Metrics:")
            print(json.dumps(cost_metrics, indent=2))
            
            # Trajectory Summary
            print("\nTrajectory Summary:")
            if hasattr(trajectory, 'output'): # If it's stored as object (not likely with sqlalchemy raw execute? usually dict or list)
                 # Actually trajectory is usually a list of messages.
                 pass
            
            # Check length
            if isinstance(trajectory, list):
                print(f"Total steps: {len(trajectory)}")
                # Print first few and last few lines/steps/tools
                for i, step in enumerate(trajectory):
                    role = step.get('role', 'unknown')
                    content = step.get('content', '')
                    tool_calls = step.get('tool_calls', [])
                    if tool_calls:
                        print(f"Step {i+1}: {role} called tools: {[t.get('function', {}).get('name') for t in tool_calls]}")
                    else:
                        # Truncate content
                        preview = (content[:100] + '...') if content and len(content) > 100 else content
                        print(f"Step {i+1}: {role} -> {preview}")
            else:
                 print("Trajectory is not a list?", type(trajectory))

    except Exception as e:
        print(f"Error analyzing {exp_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", required=True, help="Experiment ID to analyze")
    args = parser.parse_args()
    analyze(args.exp_id)
