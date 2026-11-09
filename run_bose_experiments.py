# Small script to run the Bose baseline, the complexity baseline, or their combination
# on the CDRIFT logs, evaluated with the same framework as run_approach.py.
# Edit the variables below, then run from the repo root:
#   python run_bose_experiments.py
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_bose_experiments.py
#
# Modes:
#   "bose"       – original Bose approach (J-measure or window count)
#   "complexity" – complexity measures tested with Bose's statistical tests
#   "combined"   – weighted combination of both approaches
#
# Complexity vectors and evaluation are handled by the existing functions in
# run_approach.py. Bose's original statistical tests and peak detection are reused.
# Only the logs used by the evaluation framework (atomic logs) are processed.
# Detected change points are trace indices.
#
# Outputs are written to baselines_results/<run_name>/:
# config.json, detections.csv, evaluation_measures.csv and the evaluation plots/CSVs.


import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd

import run_approach as ra
from cdrift.utils.helpers import importLog
from baselines.Bose_J.bose import (
    KSTest_2Sample_SlidingWindow_Step,
    MannWhitney_U_SlidingWindow_Step,
    detectChange_JMeasure_KS_Step,
    detectChange_JMeasure_MU_Step,
    detectChange_WC_KS_Step,
    detectChange_WC_MU_Step,
    visualInspection_Step,
    _detectChangeLocal_Step
)


# Experiment settings

MODE = "bose"                 # "bose" | "complexity" | "combined"
MODE_NAME = {"bose": "standard"}.get(MODE, MODE)
RUN_NAME = f"bose_wc_{MODE_NAME}"

TEST = "KS"                   # "KS" | "MU"
BOSE_MEASURE = "WC"             # "J" | "WC"
BOSE_WINDOW = 200
BOSE_STEP = 2
MEASURE_WINDOW = None

COMPLEXITY_WINDOW = 5
MEASURES = None  # None = all unique measures 	num_ties_in_transition_matrix; species_richness_abundance_n_gram_species_n2
ALPHA = 0.5                   # complexity weight in combined mode

N_WINDOWS = ra.DEFAULT_N_WINDOWS
LAG_WINDOW = 200
EVALUATE_NOISY = False

DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"
CACHE_DIR = os.path.join(OUT_ROOT, "_cache")

BOSE_FUNCTIONS = {
    ("J", "KS"): detectChange_JMeasure_KS_Step,
    ("J", "MU"): detectChange_JMeasure_MU_Step,
    ("WC", "KS"): detectChange_WC_KS_Step,
    ("WC", "MU"): detectChange_WC_MU_Step,
}

TESTS = {"KS": KSTest_2Sample_SlidingWindow_Step,
        "MU": MannWhitney_U_SlidingWindow_Step}


# Calculate bose and complexity p-values

def bose_pvalues(event_log, label):
    path = f"{CACHE_DIR}/{label}__{BOSE_MEASURE}_{TEST}_w{BOSE_WINDOW}_s{BOSE_STEP}_m{MEASURE_WINDOW}.npz"
    if os.path.exists(path):
        with np.load(path) as data:
            return data["p"], float(data["seconds"])
    start = time.perf_counter()
    p = _detectChangeLocal_Step(event_log, TEST, BOSE_MEASURE, BOSE_WINDOW, MEASURE_WINDOW, BOSE_STEP,
                                show_progress_bar=False)
    seconds = time.perf_counter() - start
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, p=p, seconds=seconds)
    return p, seconds
 
 
def complexity_pvalues(Z, borders, n_traces):
    grid = np.arange(0, n_traces - 2 * BOSE_WINDOW, BOSE_STEP) + BOSE_WINDOW
    columns = np.flatnonzero(Z.std(axis=0) > 1e-10)
    if len(columns) == 0:
        return np.ones(len(grid))
    p = np.mean([TESTS[TEST](Z[:, j], COMPLEXITY_WINDOW, 1) for j in columns], axis=0)
    starts = [borders[i + COMPLEXITY_WINDOW][0] for i in range(len(p))]
    return np.interp(grid, starts, p)
 
 
def detect_one(log, measures, mode, run_name):
    dataset = log["dataset"]
    name = log["log_filename"].split(".")[0]
    try:
        actual = ra.load_ground_truth(log)
        if not actual:
            return None
        p_bose, bose_seconds = None, 0.0
        if mode != "complexity":
            path = f"{DATASETS_PATH}/{dataset}/{log['log_filename']}"
            if not os.path.exists(path):
                path += ".gz"
            event_log = importLog(path, verbose=False)
            n_traces = len(event_log)
            p_bose, bose_seconds = bose_pvalues(event_log, f"{dataset}__{name}")
 
        start = time.perf_counter()
        if mode != "bose":
            folder = ra.find_vector_folder(DATASETS_PATH, dataset, name, N_WINDOWS[dataset])
            Z, windows_traces = ra.load_standardized_matrix(folder, name, measures)
            with open(f"{folder}/{name}_borders.json") as f:
                borders = json.load(f)
            if mode == "combined" and windows_traces != n_traces:
                raise ValueError(f"log has {n_traces} traces, the complexity windows cover {windows_traces}")
            n_traces = windows_traces
            p_complexity = complexity_pvalues(Z, borders, n_traces)
 
        if mode == "bose":
            p = p_bose
        elif mode == "complexity":
            p = p_complexity
        else:
            p = (1 - ALPHA) * p_bose + ALPHA * p_complexity
        peaks = visualInspection_Step(p, BOSE_WINDOW, BOSE_STEP)
        detected = sorted({min(int(c), n_traces - 1) for c in peaks})
        duration = bose_seconds + time.perf_counter() - start
    except Exception as e:
        print(f"[ERROR] {dataset}/{name}: {e!r}", flush=True)
        return None
    return {
        "Algorithm": run_name,
        "Log Source": dataset,
        "Log": name,
        "Detected Changepoints": detected,
        "Actual Changepoints for Log": actual,
        "Duration (Seconds)": duration,
        "Duration": time.strftime("%H:%M:%S", time.gmtime(duration)),
    }
 
 
def is_atomic(log):
    name = log["log_filename"].split(".")[0].lower()
    if log["dataset"] == "Ostovar":
        return name.startswith("atomic")
    if log["dataset"] == "Ceravolo":
        return set(name.split("_")[-1]) != {"i", "o", "r"}
    return True
 
 
# Run
def _init_worker(alpha):
    """Initialize worker process with the global ALPHA value."""
    global ALPHA
    ALPHA = alpha

def run(run_name, mode, measures, out_root, evaluate_noisy=EVALUATE_NOISY):
    """Runs one configuration (mode, measures; the other settings come from the variables above) and evaluates it."""
    assert mode in ("bose", "complexity", "combined") and TEST in TESTS and 0 <= ALPHA <= 1

    logs = [log for log in ra.list_logs(DATASETS_PATH) if is_atomic(log)]
    if mode == "bose":
        measures = None
    else:
        measures = measures or ra.unique_measure_names(DATASETS_PATH, N_WINDOWS)
        unknown = set(measures) - set(ra.all_measure_names(DATASETS_PATH, N_WINDOWS))
        if unknown:
            raise ValueError(f"Unknown measures: {sorted(unknown)}")
        ra.ensure_complexity_vectors(logs, N_WINDOWS, DATASETS_PATH)

    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump({"mode": mode, "test": TEST, "bose_measure": BOSE_MEASURE, "bose_window": BOSE_WINDOW,
                   "bose_step": BOSE_STEP, "measure_window": MEASURE_WINDOW, "complexity_window": COMPLEXITY_WINDOW,
                   "measures": measures, "alpha": ALPHA, "n_windows": N_WINDOWS,
                   "lag_window": LAG_WINDOW, "evaluate_noisy": evaluate_noisy}, f, indent=2)

    print(f"{run_name}: {len(logs)} logs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=None, initializer=_init_worker, initargs=(ALPHA,)) as executor:
        futures = [executor.submit(detect_one, log, measures, mode, run_name) for log in logs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is not None:
                rows.append(row)
                print(f"[{i}/{len(logs)}] {row['Log Source']}/{row['Log']} {row['Duration (Seconds)']:.0f}s", flush=True)

    detections = pd.DataFrame(rows)
    detections.to_csv(f"{out_dir}/detections.csv", index=False)

    results = ra.evaluate(detections, run_name, LAG_WINDOW, DATASETS_PATH, out_dir, evaluate_noisy)
    excluded = ("Versatility per Change Pattern", "Radar Chart", "Versatility CSV", "Versatility by Noise CSV",
                "Evaluation by Noise CSV")
    pd.DataFrame([{k: v for k, v in results.items() if k not in excluded}]).to_csv(
        f"{out_dir}/evaluation_measures.csv", index=False)
    return results


if __name__ == "__main__":
    for key, value in run(RUN_NAME, MODE, MEASURES, OUT_ROOT).items():
        print(f"{key}: {value}")
 