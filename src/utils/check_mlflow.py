import mlflow
import pandas as pd
from pathlib import Path

def check_tracking():
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    # Experiments
    try:
        exps = mlflow.search_experiments()
        print(f"Found {len(exps)} experiments:")
        for exp in exps:
            print(f" - {exp.name} (ID: {exp.experiment_id})")
            
            # Runs
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                print(f"   Runs: {len(runs)}")
                best_run = runs.iloc[0]
                metrics = [c for c in runs.columns if c.startswith("metrics.")]
                print(f"   Latest Metrics: {best_run[metrics].to_dict()}")
            else:
                print("   No runs found.")
    except Exception as e:
        print(f"Error checking mlflow: {e}")

if __name__ == "__main__":
    check_tracking()
