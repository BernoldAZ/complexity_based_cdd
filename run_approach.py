# Endpoint for the "Selin" approach: runs KernelCPD drift detection on the CDRIFT logs with
# user-chosen parameters (complexity measures, penalty and number of windows per dataset) and
# evaluates the detections with the unified evaluation framework.
#
# Detection is ported from notebooks/selin_final_pipeline.ipynb, evaluation from
# notebooks/evaluation_unified_framework.ipynb (same scoring rules, applied to this run only).
#
# Usage (from the repo root):
#   python run_approach.py --run-name run_2_measures --measures lempel_ziv_complexity variant_entropy
#   python run_approach.py --run-name run_measure_coverage --measures "coverage (n=(2,))" \
#       --n-windows Bose=75,Ceravolo=100,Ostovar=75 --pen Bose=2.0,Ceravolo=1.5,Ostovar=2.0
#   python run_approach.py --list-measures
#
# or from Python:
#   from run_approach import run
#   results = run("run_2_measures", measures=["lempel_ziv_complexity", "variant_entropy"])
#
# Outputs are written to runs/<run_name>/ (overwritten if the run name is reused).
#
# Step by step (run()):
#
#   0. Check the inputs: run name, measure names, and pen / n_windows dicts with exactly the keys
#      Bose, Ceravolo and Ostovar. Save them to config.json.
#
#   Detection (detect()), done for each of the 151 CDRIFT logs with ground truth:
#   1. Make sure complexity vectors exist for the requested number of windows N. They are read
#      from <dataset>/ or <dataset>/nw<N>/; if missing, they are computed once with
#      complexity_measures/compute_log_complexity.py (the log is cut into N windows and the
#      53 measures are computed per window) and saved in <dataset>/nw<N>/.
#   2. Load the log's matrix (N windows x 53 measures) and keep only the selected measures,
#      giving N x k (k = number of selected measures; one measure -> a single series).
#   3. Standardize each column within the log (z-score), so that all measures have the same scale.
#   4. Fit KernelCPD (RBF kernel, min_size=2; the kernel width is set per log by the median
#      heuristic) and predict change points with the penalty of the log's dataset. The result
#      is a list of window indexes. Steps 2-4 are timed as the run's duration.
#   5. Convert the window indexes to trace indexes (window index x traces per window), the unit
#      of the ground truth from drift_info.csv. All rows are saved to detections.csv.
#
#   Evaluation (evaluate()), same rules as evaluation_unified_framework.ipynb, lag window 200:
#   6. Keep only the 90 atomic logs and label each with its change pattern and noise level.
#      The noiseless subset has 24 logs.
#   7. Accuracy on the noiseless logs: TPs/FPs are summed over all logs, then Precision, Recall
#      and F1 are computed once.
#   8. Latency on the noiseless logs: mean distance (in traces) between each detected and
#      matched actual change point.
#   9. Versatility on the 23 noiseless Ceravolo/Ostovar logs: recall per change pattern, then the
#      mean over patterns. Saved as versatility_per_change_pattern.csv and as a radar chart PNG.
#  10. Scalability on the 90 atomic logs: mean duration, and milliseconds per case / per event
#      (log lengths are read from the .xes.gz files).
#  11. Parameter Sensitivity: IQR of the harmonic mean of accuracy, latency and versatility over
#      the parameter settings. A run has one setting, so this is always 0.
#  12. Robustness on the 89 atomic Ceravolo/Ostovar logs: harmonic mean of accuracy, latency and
#      versatility per noise level, area under that curve relative to the noise-0 level, and the
#      mean of Ceravolo and Ostovar.
#  13. Save all measures to evaluation_measures.csv and return them as a dict.

import argparse
import ast
import contextlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import harmonic_mean, mean

import matplotlib
matplotlib.use("Agg")  # no display needed; figures are only saved
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import iqr

# This repo's cdrift/ only ships evaluation.py and shadows the upstream package; also search the
# upstream cdrift package on PYTHONPATH so that cdrift.utils can be imported.
import pkgutil
import cdrift
cdrift.__path__ = pkgutil.extend_path(cdrift.__path__, "cdrift")
from cdrift import evaluation

DATASETS = ["Bose", "Ceravolo", "Ostovar"]
DEFAULT_N_WINDOWS = {"Bose": 75, "Ceravolo": 100, "Ostovar": 75}
DEFAULT_PEN = {"Bose": 2.0, "Ceravolo": 2.0, "Ostovar": 2.0}

COUNT_DUPLICATE_DETECTIONS = False
MIN_SUPPORT = 1
PARAM_COLUMNS = ["Run Config"]  # one constant value per run, see evaluate()


# ─── Detection ────────────────────────────────────────────────────────────────

def trace_idx_to_window_idx(trace_idx, n_traces, N):
    window_size = n_traces / N
    window_idx = math.floor(trace_idx / window_size)
    return min(window_idx, N - 1)


def list_logs(datasets_path):
    """All 151 CDRIFT logs with documented ground truth (1 Bose + 75 Ceravolo + 75 Ostovar)."""
    logs = [{"dataset": "Bose", "log_filename": "bose_log.xes",
             "drift_info_path": f"{datasets_path}/Bose/drift_info.csv"}]
    for dataset_name in ["Ceravolo", "Ostovar"]:
        folder = f"{datasets_path}/{dataset_name}"
        drift_info_path = f"{folder}/drift_info.csv"
        logs_with_ground_truth = set(pd.read_csv(drift_info_path)["log_name"].unique())
        for f in os.listdir(folder):
            if f.endswith(".xes.gz") and f[:-3] in logs_with_ground_truth:
                logs.append({"dataset": dataset_name, "log_filename": f[:-3],
                             "drift_info_path": drift_info_path})
    return logs


def all_measure_names(datasets_path):
    with open(f"{datasets_path}/Bose/bose_log_measure_names.json") as f:
        return json.load(f)


def unique_measure_names(datasets_path):
    """The measures left after dropping exact duplicates (same rule as the final pipeline)."""
    vectors = np.load(f"{datasets_path}/Bose/bose_log_complexity_vectors.npy")
    names = all_measure_names(datasets_path)
    duplicated = {
        names[j]
        for i in range(len(names)) for j in range(i + 1, len(names))
        if np.allclose(vectors[:, i], vectors[:, j])
    }
    return [n for n in names if n not in duplicated]


def find_vector_folder(datasets_path, dataset, log_name, N):
    """Folder holding this log's complexity vectors computed with N windows, or None."""
    for folder in [f"{datasets_path}/{dataset}", f"{datasets_path}/{dataset}/nw{N}"]:
        path = f"{folder}/{log_name}_complexity_vectors.npy"
        if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == N:
            return folder
    return None


def ensure_complexity_vectors(logs, n_windows, datasets_path):
    """Computes the complexity vectors for every log that has none for the requested window count.
    Needs the log_diversity_measures package; results are cached in <dataset>/nw<N>/."""
    missing = [log for log in logs
               if find_vector_folder(datasets_path, log["dataset"], log["log_filename"].split(".")[0],
                                     n_windows[log["dataset"]]) is None]
    if not missing:
        return
    print(f"Computing complexity vectors for {len(missing)} logs (this is slow)...")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "complexity_measures"))
    from compute_log_complexity import process_single_log
    # process_single_log builds its paths from the bare dataset folder name
    with contextlib.chdir(datasets_path):
        for i, log in enumerate(missing, start=1):
            N = n_windows[log["dataset"]]
            print(f"  [{i}/{len(missing)}] {log['dataset']}/{log['log_filename']} (n_windows={N})")
            process_single_log(log["dataset"], f"{log['log_filename']}.gz", i,
                               n_windows=N, output_subfolder=f"nw{N}")


def load_standardized_matrix(folder, log_name, measures):
    """Loads one log's complexity vectors, keeps only the selected measures and z-scores each
    column. Returns (standardized_matrix, n_traces)."""
    vectors = np.load(f"{folder}/{log_name}_complexity_vectors.npy")
    with open(f"{folder}/{log_name}_measure_names.json") as f:
        measure_names = json.load(f)
    with open(f"{folder}/{log_name}_borders.json") as f:
        borders = json.load(f)

    n_traces = borders[-1][1] + 1
    keep_idx = [measure_names.index(name) for name in measures]
    matrix = vectors[:, keep_idx]
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds_safe = np.where(stds < 1e-10, 1, stds)
    return (matrix - means) / stds_safe, n_traces


def load_ground_truth(log):
    drift_df = pd.read_csv(log["drift_info_path"])
    log_drifts = drift_df[drift_df["log_name"] == log["log_filename"]]
    true_trace_cps = sorted(set(
        int(t) for raw in log_drifts["drift_traces_index"] for t in ast.literal_eval(raw)
    ))
    return true_trace_cps


def detect(run_name, measures, pen, n_windows, datasets_path):
    """Runs KernelCPD on every log; returns one row per log in the master results schema."""
    logs = list_logs(datasets_path)
    ensure_complexity_vectors(logs, n_windows, datasets_path)

    rows = []
    for log in logs:
        dataset = log["dataset"]
        log_name = log["log_filename"].split(".")[0]
        N = n_windows[dataset]
        folder = find_vector_folder(datasets_path, dataset, log_name, N)

        start_time = time.time()
        standardized, n_traces = load_standardized_matrix(folder, log_name, measures)
        actual_trace_cps = load_ground_truth(log)
        if len(actual_trace_cps) == 0:
            continue
        algo = rpt.KernelCPD(kernel="rbf", min_size=2).fit(standardized)
        detected_windows = [w for w in algo.predict(pen=pen[dataset]) if w < N]
        execution_time = time.time() - start_time  # stops before scoring, which isn't part of the algorithm

        # Window index -> trace index estimate (inverse of the window-index formula)
        window_size = n_traces / N
        detected_trace_cps = sorted(set(
            min(round(w * window_size), n_traces - 1) for w in detected_windows
        ))
        rows.append({
            "Algorithm": run_name,
            "Log Source": dataset,
            "Log": log_name,
            "Pen": pen[dataset],
            "Detected Changepoints": detected_trace_cps,
            "Actual Changepoints for Log": actual_trace_cps,
            "Duration (Seconds)": execution_time,
        })
    print(f"Detection done on {len(rows)} logs")
    return pd.DataFrame(rows)


# ─── Evaluation (ported from evaluation_unified_framework.ipynb) ──────────────

def map_row_to_cp(row):
    mapping_ostovar_to_shortnames = {
        "ConditionalMove": 'cm', "ConditionalRemoval": 'cre', "ConditionalToSequence": 'cf',
        "Frequency": 'fr', "Loop": 'lp', "ParallelMove": 'pm', "ParallelRemoval": 'pre',
        "ParallelToSequence": 'pl', "SerialMove": 'sm', "SerialRemoval": 'sre', "Skip": 'cb',
        "Substitute": 'rp', "Swap": 'sw',
    }
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return logname.split("_")[-1]
    elif row["Log Source"] == "Ostovar":
        change_pattern = logname.split("_")[-1]
        if change_pattern.isnumeric():  # This log has a noise level; use the second to last string instead
            change_pattern = logname.split("_")[-2]
        return mapping_ostovar_to_shortnames[change_pattern]
    return None  # Bose


def map_row_to_noise(row):
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return re.search('noise([0-9]*)_', logname).group(1)
    elif row["Log Source"] == "Ostovar":
        last_member = logname.split('_')[-1]
        return last_member if last_member.isnumeric() else '0'
    return None  # Bose


def load_log_lengths(logpath_str):
    from cdrift.utils.helpers import importLog
    log = importLog(logpath_str, verbose=False)
    return logpath_str, len(log), sum(len(case) for case in log)


def calcAccuracy(df, param_names, lag_window):
    f1s, recalls, precisions = dict(), dict(), dict()
    for parameters, group in df.groupby(by=param_names):
        tps = fps = positives = detected = 0
        for _, row in group.iterrows():
            actual_cp = row["Actual Changepoints for Log"]
            detected_cp = row["Detected Changepoints"]
            tp, fp = evaluation.getTP_FP(detected_cp, actual_cp, lag_window, count_duplicate_detections=False)
            tps += tp
            fps += fp
            positives += len(actual_cp)
            detected += tp + fp
        precisions[parameters] = tps / detected if detected else np.nan
        recalls[parameters] = tps / positives if positives else np.nan
        f1s[parameters] = harmonic_mean([precisions[parameters], recalls[parameters]])
    return precisions, recalls, f1s


def calculate_accuracy(df, lag_window):
    precisions, recalls, f1s = calcAccuracy(df, PARAM_COLUMNS, lag_window)
    best_param = max(f1s, key=lambda x: f1s[x])
    return f1s[best_param], f1s, precisions[best_param], recalls[best_param]


def calcLatencies(df, param_names, lag_window):
    latencies = dict()
    for parameters, group in df.groupby(by=param_names):
        lags = []
        for _, row in group.iterrows():
            assignments = evaluation.assign_changepoints(
                row["Detected Changepoints"], row["Actual Changepoints for Log"], lag_window)
            lags.extend(abs(d - a) for d, a in assignments)
        latencies[parameters] = lags
    return latencies


def calculate_latency(df, lag_window, min_support=1):
    result = calcLatencies(df, PARAM_COLUMNS, lag_window)
    scaled = {
        param: 1 - (mean(result[param]) / lag_window) if len(result[param]) >= min_support else np.nan
        for param in result.keys()
    }
    best_param = max(scaled, key=lambda x: scaled[x] if not np.isnan(scaled[x]) else -1)
    return scaled[best_param], scaled


def calc_versatility(df, lag_window):
    recalls_per_param = dict()
    for params, params_group in df.groupby(by=PARAM_COLUMNS):
        recalls = dict()
        for change_pattern, cp_group in params_group.groupby(by="Change Pattern"):
            TPS = POSITIVES = 0
            for _, row in cp_group.iterrows():
                tp, _ = evaluation.getTP_FP(row["Detected Changepoints"], row["Actual Changepoints for Log"],
                                            lag_window, count_duplicate_detections=COUNT_DUPLICATE_DETECTIONS)
                TPS += tp
                POSITIVES += len(row["Actual Changepoints for Log"])
            recalls[change_pattern] = TPS / POSITIVES if POSITIVES != 0 else np.nan
        recalls_per_param[params] = recalls
    best_param = max(recalls_per_param, key=lambda x: np.nanmean(list(recalls_per_param[x].values())))
    versatility = np.nanmean(list(recalls_per_param[best_param].values()))
    return versatility, recalls_per_param, best_param


def calc_robustnesses(df, lag_window):
    """{noise_level: robustness} -- harmonic mean of accuracy, latency and versatility per noise level."""
    means = dict()
    for noise_level, noise_df in df.groupby("Noise Level"):
        accuracy, _, _, _ = calculate_accuracy(noise_df, lag_window)
        latency, _ = calculate_latency(noise_df, lag_window, min_support=MIN_SUPPORT)
        versatility, _, _ = calc_versatility(noise_df, lag_window)
        means[noise_level] = harmonic_mean([accuracy, latency, versatility])
    return means


def convert_harm_mean_to_auc(means, div_zero_default=0):
    # If any harmonic mean is nan, it counts as 0
    points = sorted(((int(noise), 0 if np.isnan(r) else r) for noise, r in means.items()), key=lambda x: x[0])
    initial_robustness = points[0][1]  # Robustness for the lowest (0%) noise
    ideal_auc = initial_robustness * (points[-1][0] - points[0][0])
    auc = 0
    prev_noise, prev_robust = points[0]
    for noise, robust in points[1:]:
        auc += ((noise - prev_noise) * (prev_robust + robust)) / 2
        prev_noise, prev_robust = noise, robust
    return auc / ideal_auc if ideal_auc != 0 else div_zero_default


def plot_radar_chart(recalls, title, output_path):
    categories = list(recalls.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close the circle
    values = [0 if np.isnan(recalls[cp]) else recalls[cp] for cp in categories]
    values += values[:1]  # close the shape

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
    ax.plot(angles, values, color="#DB444B", linewidth=2)
    ax.fill(angles, values, color="#DB444B", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title(title, va='bottom', fontsize=16)
    fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(df, run_name, lag_window, datasets_path, out_dir):
    df = df.copy()
    # Every row of a run shares one parameter setting. Grouping by "Pen" would split a run with
    # per-dataset penalties into one "setting" per dataset, so a constant column is used instead.
    df["Run Config"] = run_name

    # Only atomic logs
    df = df[
        ((df["Log Source"] == "Ostovar") & df["Log"].str.lower().str.startswith("atomic"))
        | df["Log"].str.lower().str.startswith("bose")
        | ((df["Log Source"] == "Ceravolo") & df["Log"].apply(lambda log: set(log.lower().split("_")[-1]) != {'i', 'o', 'r'}))
    ].copy()
    df["Change Pattern"] = df.apply(map_row_to_cp, axis=1)
    df["Noise Level"] = df.apply(map_row_to_noise, axis=1)

    df_noiseless = df[
        ((df["Log Source"] == "Ostovar") & ~(df["Log"].str.endswith("_2") | df["Log"].str.endswith("_5")))
        | ((df["Log Source"] == "Ceravolo") & (df["Log"].str.split("_").str[2] == "noise0"))
        | (df["Log Source"] == "Bose")
    ]

    # Log lengths (cases / events) for the per-case and per-event durations
    logpaths = [Path(datasets_path, source, f"{log}.xes.gz").as_posix()
                for source, log in df[["Log Source", "Log"]].drop_duplicates().itertuples(index=False)]
    loglengths, loglengths_events = {}, {}
    with ProcessPoolExecutor() as executor:
        for logpath_str, cases, events in executor.map(load_log_lengths, logpaths):
            loglengths[logpath_str] = cases
            loglengths_events[logpath_str] = events

    # Accuracy, Latency, Versatility (noiseless logs)
    f1, f1_per_param, precision, recall = calculate_accuracy(df_noiseless, lag_window)
    latency_score, latency_per_param = calculate_latency(df_noiseless, lag_window, min_support=MIN_SUPPORT)
    df_v = df_noiseless[df_noiseless["Log Source"].isin(["Ceravolo", "Ostovar"])]
    versatility, versatility_recalls, best_param_versatility = calc_versatility(df_v, lag_window)
    recalls_per_cp = versatility_recalls[best_param_versatility]

    # Scalability (all atomic logs)
    avg_seconds = df["Duration (Seconds)"].mean()
    log_keys = df.apply(lambda x: Path(datasets_path, x["Log Source"], x["Log"] + ".xes.gz").as_posix(), axis=1)
    total_seconds = df["Duration (Seconds)"].sum()
    seconds_per_case = total_seconds / log_keys.map(loglengths).sum()
    seconds_per_event = total_seconds / log_keys.map(loglengths_events).sum()

    # Parameter sensitivity: IQR of the harmonic mean of accuracy, latency and versatility per setting
    sensitivities = [
        harmonic_mean([f1_per_param[p], np.nanmean(list(versatility_recalls[p].values())), latency_per_param[p]])
        for p in f1_per_param.keys()
    ]
    sensitivity_iqr = iqr(sensitivities, nan_policy="omit")

    # Robustness: mean of the normalized AUC over noise levels for Ceravolo and Ostovar
    df_robust = df[df["Log Source"].isin(["Ceravolo", "Ostovar"])]
    robustness_ceravolo = convert_harm_mean_to_auc(calc_robustnesses(df_robust[df_robust["Log Source"] == "Ceravolo"], lag_window))
    robustness_ostovar = convert_harm_mean_to_auc(calc_robustnesses(df_robust[df_robust["Log Source"] == "Ostovar"], lag_window))
    robustness = (robustness_ceravolo + robustness_ostovar) / 2

    radar_path = f"{out_dir}/radar_chart_versatility.png"
    plot_radar_chart(recalls_per_cp, run_name, radar_path)
    versatility_csv_path = f"{out_dir}/versatility_per_change_pattern.csv"
    pd.DataFrame([{"Change Pattern": cp, "Recall": r} for cp, r in recalls_per_cp.items()]).to_csv(
        versatility_csv_path, index=False)

    return {
        "Algorithm": run_name,
        "F1-Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Latency": (1 - latency_score) * lag_window,
        "Latency (Score)": latency_score,
        "Versatility": versatility,
        "Scalability": time.strftime("%H:%M:%S", time.gmtime(avg_seconds)),
        "Milliseconds per Case": seconds_per_case * 1000,
        "Milliseconds per Event": seconds_per_event * 1000,
        "Parameter Sensitivity (IQR)": sensitivity_iqr,
        "Robustness": robustness,
        "Versatility per Change Pattern": recalls_per_cp,
        "Radar Chart": radar_path,
        "Versatility CSV": versatility_csv_path,
    }


# ─── Endpoint ─────────────────────────────────────────────────────────────────

def _check_dataset_dict(name, value):
    if not isinstance(value, dict) or set(value.keys()) != set(DATASETS):
        raise ValueError(f"{name} must be a dict with exactly the keys {DATASETS}, got {value!r}")


def run(run_name, measures=None, pen=DEFAULT_PEN, n_windows=DEFAULT_N_WINDOWS, lag_window=200,
        datasets_path="datasets/cdrift", out_root="runs"):
    """Runs the approach with the given parameters and evaluates it.

    Args:
        run_name: name of the run, e.g. "run_2_measures"; used as the output folder name.
        measures: complexity measure names to use; None uses all measures left after dropping exact duplicates.
        pen: KernelCPD penalty per dataset, e.g. {"Bose": 2.0, "Ceravolo": 2.0, "Ostovar": 2.0}.
        n_windows: number of windows per log, per dataset, e.g. {"Bose": 75, "Ceravolo": 100, "Ostovar": 75}.
        lag_window: lag window (in traces) of the evaluation framework.

    Returns:
        dict with the evaluation results (F1-Score, Precision, Recall, Latency, Versatility,
        Scalability, Milliseconds per Case/Event, Parameter Sensitivity (IQR), Robustness) and
        the paths of the saved files.
    """
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_name):
        raise ValueError(f"run_name may only contain letters, digits, '_' and '-', got {run_name!r}")
    _check_dataset_dict("pen", pen)
    _check_dataset_dict("n_windows", n_windows)
    if measures is None:
        measures = unique_measure_names(datasets_path)
    valid = all_measure_names(datasets_path)
    unknown = [m for m in measures if m not in valid]
    if unknown or not measures:
        raise ValueError(f"Unknown or empty measures {unknown}. Valid measures:\n  " + "\n  ".join(valid))

    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    config = {"run_name": run_name, "measures": list(measures), "pen": pen, "n_windows": n_windows,
              "lag_window": lag_window}
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    detections = detect(run_name, measures, pen, n_windows, datasets_path)
    detections.to_csv(f"{out_dir}/detections.csv", index=False)

    results = evaluate(detections, run_name, lag_window, datasets_path, out_dir)
    table_columns = [k for k in results if k not in ("Versatility per Change Pattern", "Radar Chart", "Versatility CSV")]
    pd.DataFrame([{k: results[k] for k in table_columns}]).to_csv(f"{out_dir}/evaluation_measures.csv", index=False)
    results["Output Folder"] = out_dir
    return results


def _parse_dataset_dict(text, cast):
    """'Bose=75,Ceravolo=100,Ostovar=75' -> {'Bose': 75, 'Ceravolo': 100, 'Ostovar': 75}"""
    try:
        return {k.strip(): cast(v) for k, v in (item.split("=") for item in text.split(","))}
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected Bose=..,Ceravolo=..,Ostovar=.., got {text!r}")


def _format_dataset_dict(d):
    return ",".join(f"{k}={v}" for k, v in d.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the KernelCPD complexity-measure approach.")
    parser.add_argument("--run-name", help='name of the run, e.g. "run_2_measures"')
    parser.add_argument("--measures", nargs="+", default=None,
                        help="complexity measures to use (default: all measures left after dropping exact duplicates)")
    parser.add_argument("--pen", type=lambda s: _parse_dataset_dict(s, float), default=DEFAULT_PEN,
                        help=f"KernelCPD penalty per dataset (default: {_format_dataset_dict(DEFAULT_PEN)})")
    parser.add_argument("--n-windows", type=lambda s: _parse_dataset_dict(s, int), default=DEFAULT_N_WINDOWS,
                        help=f"number of windows per dataset (default: {_format_dataset_dict(DEFAULT_N_WINDOWS)})")
    parser.add_argument("--lag-window", type=int, default=200)
    parser.add_argument("--datasets-path", default="datasets/cdrift")
    parser.add_argument("--list-measures", action="store_true", help="print the valid measure names and exit")
    args = parser.parse_args()

    if args.list_measures:
        print("\n".join(all_measure_names(args.datasets_path)))
        sys.exit(0)
    if not args.run_name:
        parser.error("--run-name is required")

    results = run(args.run_name, measures=args.measures, pen=args.pen, n_windows=args.n_windows,
                  lag_window=args.lag_window, datasets_path=args.datasets_path)
    print()
    for key, value in results.items():
        print(f"{key}: {value}")
