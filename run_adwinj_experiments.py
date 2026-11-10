# Small script to run the ADWIN/J baseline (Martjushev et al.), the complexity measures, or their combination
# on the CDRIFT logs, evaluated with the same framework as run_approach.py.
# Edit the variables below, then run from the repo root:  python run_adwinj_experiments.py
# Other scripts (sweeps, patched runs) set the module variables, e.g. ALPHA, and call run().
#
# Modes:
#   "adwin"       original ADWIN/J: J-measure of every activity pair, average p-value over the pairs
#   "complexity"  the complexity measures only
#   "combined"    p = (1 - ALPHA) * p_J + ALPHA * p_C, computed inside every ADWIN step
#
# ADWIN decides inside its loop, so there is no common p-value series to mix afterwards. Instead the p-value
# function that ADWIN calls on every pair of populations is replaced: ADWIN gets a one-row signal with the trace
# indices and a testing function that looks up the data of the two populations. The ADWIN code is used unchanged
# (adaptive windows, threshold, recursive bisection). ALPHA = 0 gives ADWIN/J and ALPHA = 1 the complexity measures only.
#
#   p_J = mean over all activity pairs of the test p-value of the two populations (as in the baseline)
#   p_C = mean over the complexity measures of the test p-value of the windows (sublogs) that lie completely
#         inside each population. With fewer than 2 windows in a population p_C = 1 (no evidence).
#
# In "combined" mode the J series are the slow part and do not depend on the ADWIN settings, so they are cached in
# CACHE_DIR together with their computation time (added to the duration of the log).
# Delete the folder to recompute. The "adwin" mode calls the baseline function directly
# and does not use the cache.
#
# Outputs are written to <out_root>/<run_name>/: config.json, detections.csv, evaluation_measures.csv and the
# evaluation plots / CSVs.

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.stats as stats

import run_approach as ra  
from cdrift.utils.helpers import importLog
from constants import ALL_MEASURES
from baselines.Adwin_J.martjushev import (
    _applyAvgPVal,
    _extractAllJMeasures,
    detectChange_AvgSeries_ADWIN,
    detectChange_ADWIN_JMeasure_KS,
    detectChange_ADWIN_JMeasure_MU,
)

# Settings 

MODE = "adwin"                  # "adwin" | "complexity" | "combined"
MODE_NAME = {"adwin": "standard"}.get(MODE, MODE)
RUN_NAME = f"adwin_{MODE_NAME}"

TEST = "KS"                     # "KS" | "MU"
THRESHOLD = 0.4                 # p-value threshold
MIN_WINDOW = 100                # in traces
MAX_WINDOW = 400                # in traces
STEP = 50                       # in traces
MEASURE_WINDOW = None           # window of the J measure itself, None = average trace length

MEASURES = ALL_MEASURES        
ALPHA = 0.5                     # complexity weight in combined mode

N_WINDOWS = ra.DEFAULT_N_WINDOWS
LAG_WINDOW = 200
EVALUATE_NOISY = False
MAX_WORKERS = None              # None = all cores, lower it if memory runs out

DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"
CACHE_DIR = os.path.join(OUT_ROOT, "_cache")

TESTS = {"KS": stats.ks_2samp, "MU": stats.mannwhitneyu}
ADWIN_J = {"KS": detectChange_ADWIN_JMeasure_KS, "MU": detectChange_ADWIN_JMeasure_MU}


# Detection 

def j_signals(event_log, label, cfg):
    path = f"{cfg['cache_dir']}/{label}__J_signals_m{cfg['measure_window']}.npz"
    if os.path.exists(path):
        with np.load(path) as data:
            return data["signals"], float(data["seconds"])
    start = time.perf_counter()
    signals = _extractAllJMeasures(event_log, cfg["measure_window"], show_progress_bar=False)
    seconds = time.perf_counter() - start
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    np.savez(path, signals=signals, seconds=seconds)
    return signals, seconds


def make_p_value(J, Z, starts, ends, alpha, test):
    """p-value function for ADWIN. It gets the trace indices of two populations and returns (1 - alpha) * p_J + alpha * p_C."""
    def p_value(idx1, idx2):
        if len(idx1) == 0 or len(idx2) == 0:
            return 1.0
        s1, e1, s2, e2 = int(idx1[0]), int(idx1[-1]) + 1, int(idx2[0]), int(idx2[-1]) + 1
        p_j = _applyAvgPVal(J[:, s1:e1], J[:, s2:e2], test) if alpha < 1 else 0.0
        p_c = 0.0
        if alpha > 0:
            w1 = np.flatnonzero((starts >= s1) & (ends < e1))
            w2 = np.flatnonzero((starts >= s2) & (ends < e2))
            enough = min(len(w1), len(w2)) >= 2 and Z.shape[1] > 0
            p_c = _applyAvgPVal(Z[w1].T, Z[w2].T, test) if enough else 1.0
        return (1 - alpha) * p_j + alpha * p_c
    return p_value


def detect_one(log, measures, mode, run_name, cfg):
    dataset = log["dataset"]
    name = log["log_filename"].split(".")[0]
    try:
        actual = ra.load_ground_truth(log)
        if not actual:
            return None
        event_log = None
        if mode != "complexity":
            path = f"{cfg['datasets_path']}/{dataset}/{log['log_filename']}"
            if not os.path.exists(path):
                path += ".gz"
            event_log = importLog(path, verbose=False)
            n_traces = len(event_log)

        test = TESTS[cfg["test"]]
        j_seconds = 0.0
        if mode == "adwin":
            start = time.perf_counter()
            points = ADWIN_J[cfg["test"]](event_log, cfg["min_window"], cfg["max_window"], cfg["threshold"], cfg["step"],
                                          measure_window=cfg["measure_window"], show_progress_bar=False)
        else:
            J = None
            if mode == "combined":
                J, j_seconds = j_signals(event_log, f"{dataset}__{name}", cfg)
            start = time.perf_counter()
            folder = ra.find_vector_folder(cfg["datasets_path"], dataset, name, cfg["n_windows"][dataset])
            Z, windows_traces = ra.load_standardized_matrix(folder, name, measures)
            with open(f"{folder}/{name}_borders.json") as f:
                borders = json.load(f)
            if mode == "combined" and windows_traces != n_traces:
                raise ValueError(f"log has {n_traces} traces, the complexity windows cover {windows_traces}")
            n_traces = windows_traces
            Z = Z[:, Z.std(axis=0) > 1e-10]
            alpha = 1.0 if mode == "complexity" else cfg["alpha"]
            p_value = make_p_value(J, Z, np.array([b[0] for b in borders]), np.array([b[1] for b in borders]), alpha, test)
            points = detectChange_AvgSeries_ADWIN(np.arange(n_traces)[None, :], cfg["min_window"], cfg["max_window"],
                                                  cfg["threshold"], cfg["step"], p_value, show_progress_bar=False)
        detected = sorted({min(int(p), n_traces - 1) for p in points})
        duration = j_seconds + time.perf_counter() - start
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

def run(run_name, mode, measures, out_root, evaluate_noisy=EVALUATE_NOISY):
    """Runs one configuration (mode, measures; the other settings are read from the variables above at call time)."""
    assert mode in ("adwin", "complexity", "combined") and TEST in TESTS and 0 <= ALPHA <= 1

    logs = [log for log in ra.list_logs(DATASETS_PATH) if is_atomic(log)]
    if mode == "adwin":
        measures = None
    else:
        measures = measures or ra.unique_measure_names(DATASETS_PATH, N_WINDOWS)
        unknown = set(measures) - set(ra.all_measure_names(DATASETS_PATH, N_WINDOWS))
        if unknown:
            raise ValueError(f"Unknown measures: {sorted(unknown)}")
        ra.ensure_complexity_vectors(logs, N_WINDOWS, DATASETS_PATH)

    alpha = {"adwin": 0.0, "complexity": 1.0}.get(mode, ALPHA)  # ALPHA only matters in "combined" mode
    cfg = {"test": TEST, "threshold": THRESHOLD, "min_window": MIN_WINDOW, "max_window": MAX_WINDOW, "step": STEP,
           "measure_window": MEASURE_WINDOW, "alpha": alpha, "n_windows": N_WINDOWS,
           "datasets_path": DATASETS_PATH, "cache_dir": CACHE_DIR}
    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump({"mode": mode, "measures": measures, "lag_window": LAG_WINDOW, "evaluate_noisy": evaluate_noisy,
                   **cfg}, f, indent=2)

    print(f"{run_name}: {len(logs)} logs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(detect_one, log, measures, mode, run_name, cfg) for log in logs]
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