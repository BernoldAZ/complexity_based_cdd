# Runs ADWIN/J (Martjushev et al.), the complexity measures with the same ADWIN algorithm, or both combined
# on the CDRIFT logs, evaluated with run_approach.py. Edit the settings below, then run:  python run_adwinj_experiments.py
#
# MODE: "adwin" (J series per activity pair, over traces), "complexity" (one series per measure, over windows,
#       window sizes counted in windows) or "combined" (both detectors run separately, change points merged).
# COMBINE: "union" (a complexity point is dropped if within MERGE_DISTANCE traces of a J point) or
#          "intersection" (J points with a complexity point within MERGE_DISTANCE traces).
#
# Each detector has its own windows and threshold. ADWIN flags a change when the average p-value falls below the
# threshold. Measures that do not react stay near 0.5, so with many measures use about 0.4 to 0.45, with one measure
# 0.01 to 0.05. Retune it when MEASURES changes.
#
# A complexity change point is the last trace of the window where ADWIN locates the change, like ADWIN/J.
# The J series are slow, so they are cached in <OUT_ROOT>/_cache/ with their computation time (added to the log's
# duration). They need disk space, delete the folder to recompute.
# Outputs go to baselines_results/<run_name>/ (config, detections, evaluation files).

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.stats as stats

import run_approach as ra  # imported first, so that "cdrift" resolves to this repo's package
from cdrift.utils.helpers import importLog

from baselines.Adwin_J.martjushev import _extractAllJMeasures, detectChange_AvgSeries_ADWIN

# ─── Settings ─────────────────────────────────────────────────────────────────

MODE = "adwin"                  # "adwin" | "complexity" | "combined"
COMBINE = "union"               # "union" | "intersection" (only for "combined")
TEST = "KS"                     # "KS" | "MU"

THRESHOLD = 0.4
MIN_WINDOW = 100
MAX_WINDOW = 400
STEP = 50
MEASURE_WINDOW = None           # window of the J measure itself, None = average trace length

# Complexity measures, windows in windows (sublogs)
COMPLEXITY_THRESHOLD = 0.05     # one measure: about 0.01 to 0.05. Many measures: about 0.4 to 0.45
COMPLEXITY_MIN_WINDOW = 5
COMPLEXITY_MAX_WINDOW = 20
COMPLEXITY_STEP = 1
MEASURES = ["variant_entropy"]  # None = all unique measures

MERGE_DISTANCE = 200            # in traces, for "combined"
N_WINDOWS = ra.DEFAULT_N_WINDOWS
LAG_WINDOW = 200
EVALUATE_NOISY = False
MAX_WORKERS = None              
DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"
CACHE_DIR = f"{OUT_ROOT}/_cache"

TESTS = {"KS": stats.ks_2samp, "MU": stats.mannwhitneyu}

tag = [MODE, TEST]
if MODE != "complexity":
    tag += [f"th{THRESHOLD}", f"w{MIN_WINDOW}-{MAX_WINDOW}", f"s{STEP}"]
if MODE != "adwin":
    tag += [f"cth{COMPLEXITY_THRESHOLD}", f"cw{COMPLEXITY_MIN_WINDOW}-{COMPLEXITY_MAX_WINDOW}",
            MEASURES[0] if MEASURES and len(MEASURES) == 1 else f"{len(MEASURES) if MEASURES else 'all'}measures"]
if MODE == "combined":
    tag.append(COMBINE)
RUN_NAME = re.sub(r"[^A-Za-z0-9_\-]", "_", "adwinj_" + "_".join(tag))


# Detection

def j_signals(event_log, label):
    path = f"{CACHE_DIR}/{label}__J_signals_m{MEASURE_WINDOW}.npz"
    if os.path.exists(path):
        with np.load(path) as data:
            return data["signals"], float(data["seconds"])
    start = time.perf_counter()
    signals = _extractAllJMeasures(event_log, MEASURE_WINDOW, show_progress_bar=False)
    seconds = time.perf_counter() - start
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, signals=signals, seconds=seconds)
    return signals, seconds


def detect_one(log, measures):
    dataset = log["dataset"]
    name = log["log_filename"].split(".")[0]
    try:
        actual = ra.load_ground_truth(log)
        if not actual:
            return None
        points_j, points_c, seconds = [], [], 0.0
        if MODE != "complexity":
            path = f"{DATASETS_PATH}/{dataset}/{log['log_filename']}"
            if not os.path.exists(path):
                path += ".gz"
            event_log = importLog(path, verbose=False)
            n_traces = len(event_log)
            signals, seconds = j_signals(event_log, f"{dataset}__{name}")
            start = time.perf_counter()
            points_j = detectChange_AvgSeries_ADWIN(signals, MIN_WINDOW, MAX_WINDOW, THRESHOLD, STEP, TESTS[TEST],
                                                    show_progress_bar=False)
            seconds += time.perf_counter() - start

        start = time.perf_counter()
        if MODE != "adwin":
            folder = ra.find_vector_folder(DATASETS_PATH, dataset, name, N_WINDOWS[dataset])
            Z, windows_traces = ra.load_standardized_matrix(folder, name, measures)
            with open(f"{folder}/{name}_borders.json") as f:
                borders = json.load(f)
            if MODE == "combined" and windows_traces != n_traces:
                raise ValueError(f"log has {n_traces} traces, the complexity windows cover {windows_traces}")
            n_traces = windows_traces
            columns = np.flatnonzero(Z.std(axis=0) > 1e-10)
            if len(columns) > 0:
                found = detectChange_AvgSeries_ADWIN(Z[:, columns].T, COMPLEXITY_MIN_WINDOW, COMPLEXITY_MAX_WINDOW,
                                                     COMPLEXITY_THRESHOLD, COMPLEXITY_STEP, TESTS[TEST],
                                                     show_progress_bar=False)
                points_c = [borders[min(max(int(w), 0), len(borders) - 1)][1] for w in found]

        if MODE == "adwin":
            points = points_j
        elif MODE == "complexity":
            points = points_c
        elif COMBINE == "union":
            points = points_j + [c for c in points_c if all(abs(c - j) > MERGE_DISTANCE for j in points_j)]
        else:
            points = [j for j in points_j if any(abs(j - c) <= MERGE_DISTANCE for c in points_c)]
        detected = sorted({min(int(p), n_traces - 1) for p in points})
        duration = seconds + time.perf_counter() - start
    except Exception as e:
        print(f"[ERROR] {dataset}/{name}: {e!r}", flush=True)
        return None
    return {
        "Algorithm": RUN_NAME,
        "Log Source": dataset,
        "Log": name,
        "Detected Changepoints": detected,
        "Actual Changepoints for Log": actual,
        "Duration (Seconds)": duration,
    }


def is_atomic(log):
    name = log["log_filename"].split(".")[0].lower()
    if log["dataset"] == "Ostovar":
        return name.startswith("atomic")
    if log["dataset"] == "Ceravolo":
        return set(name.split("_")[-1]) != {"i", "o", "r"}
    return True


# Run

if __name__ == "__main__":
    assert MODE in ("adwin", "complexity", "combined") and TEST in TESTS and COMBINE in ("union", "intersection")

    logs = [log for log in ra.list_logs(DATASETS_PATH) if is_atomic(log)]
    measures = None
    if MODE != "adwin":
        measures = MEASURES or ra.unique_measure_names(DATASETS_PATH, N_WINDOWS)
        unknown = set(measures) - set(ra.all_measure_names(DATASETS_PATH, N_WINDOWS))
        if unknown:
            raise ValueError(f"Unknown measures: {sorted(unknown)}")
        ra.ensure_complexity_vectors(logs, N_WINDOWS, DATASETS_PATH)

    out_dir = f"{OUT_ROOT}/{RUN_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump({"mode": MODE, "combine": COMBINE, "test": TEST, "threshold": THRESHOLD, "min_window": MIN_WINDOW,
                   "max_window": MAX_WINDOW, "step": STEP, "measure_window": MEASURE_WINDOW,
                   "complexity_threshold": COMPLEXITY_THRESHOLD, "complexity_min_window": COMPLEXITY_MIN_WINDOW,
                   "complexity_max_window": COMPLEXITY_MAX_WINDOW, "complexity_step": COMPLEXITY_STEP,
                   "measures": measures, "merge_distance": MERGE_DISTANCE, "n_windows": N_WINDOWS,
                   "lag_window": LAG_WINDOW, "evaluate_noisy": EVALUATE_NOISY}, f, indent=2)

    print(f"{RUN_NAME}: {len(logs)} logs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(detect_one, log, measures) for log in logs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is not None:
                rows.append(row)
                print(f"[{i}/{len(logs)}] {row['Log Source']}/{row['Log']} {row['Duration (Seconds)']:.0f}s", flush=True)

    detections = pd.DataFrame(rows)
    detections.to_csv(f"{out_dir}/detections.csv", index=False)

    results = ra.evaluate(detections, RUN_NAME, LAG_WINDOW, DATASETS_PATH, out_dir, EVALUATE_NOISY)
    excluded = ("Versatility per Change Pattern", "Radar Chart", "Versatility CSV", "Versatility by Noise CSV",
                "Evaluation by Noise CSV")
    pd.DataFrame([{k: v for k, v in results.items() if k not in excluded}]).to_csv(
        f"{out_dir}/evaluation_measures.csv", index=False)
    for key, value in results.items():
        print(f"{key}: {value}")