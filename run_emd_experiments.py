# Small script to run one experiment with the Earth Mover's Distance baseline (baselines/EMD/earthmover.py)
# on the CDRIFT logs, evaluated with the same evaluation framework as run_cluster_experiments.py
# (run_approach.evaluate()). Edit the variables below, then run from the repo root:  python run_emd_experiments.py
#
# With Docker (from the repo root, in PowerShell):
#   docker build -t cbcdd-image .                                                  # once
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_emd_experiments.py
#
# Signal the change points are detected on (FEATURE_MODE), see baselines/EMD/emd_complexity_approach.py:
#   "standard":   EMD between the two adjacent windows (the original approach)
#   "combined":   EMD + distance of the complexity vectors of MEASURES of the two windows, combined with FUSION:
#                 "weighted_concat" (ALPHA weights the complexity part) or "concat" (no weighting, ALPHA unused)
#   "complexity": distance of the complexity vectors of MEASURES only
#
# One run uses one window size. The reference results (baselines/version-clustering-cdd-main/
# algorithm_results_cdrift.csv, "Earth Mover's Distance") used window sizes 100-600 with step size 2;
# 300 had the best F1. Only the logs that run_approach.evaluate() keeps (atomic logs) are processed, since EMD
# is slow. The logs are read from the original .xes.gz files, as in the reference. Detected change points are
# trace indices.
#
# The complexity vectors are computed on every EMD window and cached in
# <DATASETS_PATH>/<dataset>/emd_w<WINDOW_SIZE>_s<STEP_SIZE>/ (slow the first time). The duration of a log
# covers EMD, fusion and peak detection; computing/loading the complexity vectors is in
# "Complexity Duration (Seconds)".
#
# Outputs are written to baselines_results/<run_name>/ (overwritten if the run name is reused):
# config.json, detections.csv, evaluation_measures.csv and the versatility CSVs / radar chart.

import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from scipy.signal._peak_finding_utils import PeakPropertyWarning  # not exported by scipy.signal in 1.15

# find_peaks warns about peaks of width 0; they are filtered out by the width threshold anyway
warnings.filterwarnings("ignore", category=PeakPropertyWarning)

import run_approach as ra  # imported first, so that "cdrift" resolves to this repo's package
from cdrift.utils.helpers import importLog
from constants import ALL_MEASURES

sys.path.append(str(Path(__file__).resolve().parent / "baselines" / "EMD"))
import emd_complexity_approach as eca


# ─── Inputs of the run ────────────────────────────────────────────────────────
# Note: RUN_NAME is used as the output folder name, so it can only contain letters, digits, '_' and '-'.

#MEASURES =  ALL_MEASURES # or ALL_MEASURES; not used for "standard"
MEASURES = [
    "avg_affinity",
    "deviation_from_random",
    "enriched_normalized_variant_entropy",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3",
    "exp_shannon_entropy_abundance_n_gram_species_n2",
    "exp_shannon_entropy_abundance_n_gram_species_n3",
    "lempel_ziv_complexity",
    "normalized_variant_entropy",
    "num_events",
    "simpson_diversity_abundance_n_gram_species_n2",
    "simpson_diversity_abundance_n_gram_species_n3",
    "species_richness_abundance_n_gram_species_n3",
    "trace_length_average",
    "variant_entropy",
]

RUN_NAME = "emd_combined_CRE"
FEATURE_MODE = "combined"       # "standard" | "combined" | "complexity"
FUSION = "weighted_concat"      # "weighted_concat" | "concat" (only for "combined") 
ALPHA = 0.5                     # weight of the complexity part in [0, 1] (only for "combined") A higher ALPHA gives the complexity part more weight and the EMD part less
WINDOW_SIZE = 300               # reference: 100 | 200 | 300 | 400 | 500 | 600
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


def detect_one(log, run_name, feature_mode, fusion, alpha, measures, window_size, step_size, datasets_path):
    """Runs EMD on one log; returns its row of detections.csv, or None if the log is skipped."""
    dataset = log["dataset"]
    log_name = log["log_filename"].split(".")[0]
    try:
        actual_trace_cps = ra.load_ground_truth(log)
        if len(actual_trace_cps) == 0:
            return None
        event_log = importLog(f"{datasets_path}/{dataset}/{log['log_filename']}.gz", verbose=False)

        start_time = time.time()
        complexity = None
        if feature_mode != "standard":
            cache_folder = f"{datasets_path}/{dataset}/emd_w{window_size}_s{step_size}"
            complexity = eca.window_complexity_vectors(event_log, window_size, step_size, cache_folder, log_name)
        complexity_time = time.time() - start_time

        start_time = time.time()
        detected_trace_cps = eca.detect_change(event_log, window_size, step_size, feature_mode, fusion, alpha,
                                               measures, complexity)
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
        "Complexity Duration (Seconds)": complexity_time,
    }


def run(run_name, feature_mode="standard", fusion="weighted_concat", alpha=0.5, measures=None, window_size=300,
        step_size=2, lag_window=200, datasets_path="datasets/cdrift", out_root="baselines_results",
        evaluate_noisy=False):
    """Runs the EMD baseline with the given parameters and evaluates it like run_approach.run()."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_name):
        raise ValueError(f"run_name may only contain letters, digits, '_' and '-', got {run_name!r}")
    if feature_mode != "standard":
        unknown = [m for m in measures or [] if m not in ALL_MEASURES]
        if unknown:
            raise ValueError(f"Unknown measures {unknown}. Valid measures:\n  " + "\n  ".join(ALL_MEASURES))
    eca.check_params(feature_mode, fusion, alpha, measures)

    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    config = {"run_name": run_name, "approach": eca.algorithm_name(feature_mode, fusion),
              "feature_mode": feature_mode,
              "fusion": fusion if feature_mode == "combined" else None,
              "alpha": alpha if feature_mode == "combined" else None,
              "measures": list(measures) if feature_mode != "standard" else None,
              "window_size": window_size, "step_size": step_size, "lag_window": lag_window,
              "evaluate_noisy": evaluate_noisy}
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Detection
    logs = [log for log in ra.list_logs(datasets_path) if is_evaluated_log(log)]
    print(f"Running EMD on {len(logs)} logs")
    detect = partial(detect_one, run_name=run_name, feature_mode=feature_mode, fusion=fusion, alpha=alpha,
                     measures=measures, window_size=window_size, step_size=step_size, datasets_path=datasets_path)
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
        feature_mode=FEATURE_MODE,
        fusion=FUSION,
        alpha=ALPHA,
        measures=MEASURES,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        lag_window=LAG_WINDOW,
        datasets_path=DATASETS_PATH,
        out_root=OUT_ROOT,
        evaluate_noisy=EVALUATE_NOISY,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
