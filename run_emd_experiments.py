# Small script to run one experiment with the Earth Mover's Distance baseline (baselines/EMD/earthmover.py)
# on the CDRIFT logs, evaluated with the same evaluation framework as run_cluster_experiments.py
# (run_approach.evaluate()). Edit the variables below, then run from the repo root:  python run_emd_experiments.py
#
# With Docker (from the repo root, in PowerShell):
#   docker build -t cbcdd-image .                                                  # once
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_emd_experiments.py
#
# One run uses one window size. The reference results (baselines/version-clustering-cdd-main/
# algorithm_results_cdrift.csv, "Earth Mover's Distance") used window sizes 100-600 with step size 2.
# Only the logs that run_approach.evaluate() keeps (atomic logs) are processed, since EMD is slow.
# The logs are read from the original .xes.gz files, as in the reference. Detected change points are trace indices.
#
# Outputs are written to baselines_results/<run_name>/ (overwritten if the run name is reused):
# config.json, detections.csv, evaluation_measures.csv and the versatility CSVs / radar chart.

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

import run_approach as ra  # imported first, so that "cdrift" resolves to this repo's package
from cdrift.utils.helpers import importLog

sys.path.append(str(Path(__file__).resolve().parent / "baselines" / "EMD"))
import earthmover as emd


# ─── Inputs of the run ────────────────────────────────────────────────────────
# Note: RUN_NAME is used as the output folder name, so it can only contain letters, digits, '_' and '-'.

RUN_NAME = "emd_standard"
WINDOW_SIZE = 100               # reference: 100 | 200 | 300 | 400 | 500 | 600
STEP_SIZE = 2                   # reference: 2
LAG_WINDOW = 200
EVALUATE_NOISY = False
DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"


def is_evaluated_log(log):
    """Same atomic log filter as the start of run_approach.evaluate()."""
    log_name = log["log_filename"].split(".")[0].lower()
    return ((log["dataset"] == "Ostovar" and log_name.startswith("atomic"))
            or log_name.startswith("bose")
            or (log["dataset"] == "Ceravolo" and set(log_name.split("_")[-1]) != {'i', 'o', 'r'}))


def detect_one(log, run_name, window_size, step_size, datasets_path):
    """Runs EMD on one log; returns its row of detections.csv, or None if the log is skipped."""
    dataset = log["dataset"]
    log_name = log["log_filename"].split(".")[0]
    try:
        actual_trace_cps = ra.load_ground_truth(log)
        if len(actual_trace_cps) == 0:
            return None
        event_log = importLog(f"{datasets_path}/{dataset}/{log['log_filename']}.gz", verbose=False)

        start_time = time.time()
        detected_trace_cps = emd.detect_change(event_log, window_size, step_size, show_progress_bar=False)
        execution_time = time.time() - start_time
    except Exception as e:
        print(f"Error processing {dataset}/{log_name}: {e!r}")
        return None
    print(f"Done {dataset}/{log_name} in {execution_time:.0f}s")
    # Same columns as algorithm_results_cdrift.csv
    return {
        "Algorithm": run_name,
        "Log Source": dataset,
        "Log": log_name,
        "Detected Changepoints": [int(cp) for cp in detected_trace_cps],
        "Actual Changepoints for Log": actual_trace_cps,
        "Duration (Seconds)": execution_time,
        "Duration": time.strftime("%H:%M:%S", time.gmtime(execution_time)),
        "Window Size": window_size,
        "SW Step Size": step_size,
    }


def run(run_name, window_size=100, step_size=2, lag_window=200, datasets_path="datasets/cdrift",
        out_root="baselines_results", evaluate_noisy=False):
    """Runs the EMD baseline with the given parameters and evaluates it like run_approach.run()."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_name):
        raise ValueError(f"run_name may only contain letters, digits, '_' and '-', got {run_name!r}")

    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    config = {"run_name": run_name, "approach": "Earth Mover's Distance", "window_size": window_size,
              "step_size": step_size, "lag_window": lag_window, "evaluate_noisy": evaluate_noisy}
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Detection
    logs = [log for log in ra.list_logs(datasets_path) if is_evaluated_log(log)]
    print(f"Running EMD on {len(logs)} logs")
    detect = partial(detect_one, run_name=run_name, window_size=window_size, step_size=step_size,
                     datasets_path=datasets_path)
    with ProcessPoolExecutor() as executor:
        rows = [row for row in executor.map(detect, logs) if row is not None]
    print(f"Detection done on {len(rows)} logs")
    detections = pd.DataFrame(rows)
    detections.to_csv(f"{out_dir}/detections.csv", index=False)

    # Evaluation (same as run_approach.run)
    results = ra.evaluate(detections, run_name, lag_window, datasets_path, out_dir, evaluate_noisy)
    table_columns = [k for k in results if k not in ("Versatility per Change Pattern", "Radar Chart", "Versatility CSV", "Versatility by Noise CSV", "Evaluation by Noise CSV")]
    pd.DataFrame([{k: results[k] for k in table_columns}]).to_csv(f"{out_dir}/evaluation_measures.csv", index=False)
    results["Output Folder"] = out_dir
    return results


if __name__ == "__main__":
    results = run(
        RUN_NAME,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        lag_window=LAG_WINDOW,
        datasets_path=DATASETS_PATH,
        out_root=OUT_ROOT,
        evaluate_noisy=EVALUATE_NOISY,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
