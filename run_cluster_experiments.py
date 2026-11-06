# Small script to run one experiment with the version clustering approach
# (baselines/version-clustering-cdd-main/utilities/cluster_complexity_approach.py) on the CDRIFT logs,
# evaluated with the same evaluation framework as run_experiments.py (run_approach.evaluate()).
# Edit the variables below, then run from the repo root:  python run_cluster_experiments.py
#
# With Docker (from the repo root, in PowerShell):
#   docker build -t cbcdd-image .                                                  # once
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_cluster_experiments.py
#
# Input feature of the clustering (FEATURE_MODE):
#   "similarity": windowed DFG similarity vectors only (the original approach)
#   "combined":   similarity vectors + complexity vectors of MEASURES, combined with FUSION:
#                 "weighted_concat" (ALPHA weights the complexity part) or "concat" (no weighting, ALPHA unused)
#   "complexity": complexity vectors of MEASURES only
#
# Complexity vectors are read from <DATASETS_PATH>/<dataset>/ or <dataset>/nw<N>/ and computed once
# if missing (see run_approach.ensure_complexity_vectors).
#
# Detected window indexes are converted to trace ids exactly as the baseline does
# (cluster_approach.get_changepoints_trace_idx: case id of the first trace of window_info[w]). The duration of a
# log covers the DFG/similarity computation, clustering and change point detection (not reading the
# log or loading the complexity vectors).
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

# pm4py's DFG discovery (called once per window) raises this harmless warning on every call
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import run_approach as ra  # imported first, so that "cdrift" resolves to this repo's package
from run_experiments import ALL_MEASURES

# The baseline is imported as the "utilities" package of its own folder
sys.path.append(str(Path(__file__).resolve().parent / "baselines" / "version-clustering-cdd-main"))
import utilities.utilities as utils
import utilities.cluster_approach as ca
import utilities.cluster_complexity_approach as cca

# All 53 complexity measures that can be used.
ALL_MEASURES = [
    # Log-level measures
    "average_distinct_events_per_trace",
    "avg_affinity",
    "deviation_from_random",
    "enriched_normalized_sequence_entropy",
    "enriched_normalized_variant_entropy",
    "enriched_sequence_entropy",
    "enriched_variant_entropy",
    "lempel_ziv_complexity",
    "normalized_sequence_entropy",
    "normalized_variant_entropy",
    "num_event_types",
    "num_events",
    "num_sequences",
    "num_ties_in_transition_matrix",
    "pct_unique_sequences",
    "sequence_entropy",
    "trace_length_average",
    "trace_length_max",
    "trace_length_min",
    "variant_entropy",
    # Measures on 2-activity sequences
    "coverage_n_gram_species_n2",
    "coverage_alternative_n_gram_species_n2",
    "estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n2",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2",
    "estimate_asymptotic_species_richness_abundance_n_gram_species_n2",
    "exp_shannon_entropy_abundance_n_gram_species_n2",
    "simpson_diversity_abundance_n_gram_species_n2",
    "species_richness_abundance_n_gram_species_n2",
    # Measures on 3-activity sequences
    "coverage_n_gram_species_n3",
    "coverage_alternative_n_gram_species_n3",
    "estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n3",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3",
    "estimate_asymptotic_species_richness_abundance_n_gram_species_n3",
    "exp_shannon_entropy_abundance_n_gram_species_n3",
    "simpson_diversity_abundance_n_gram_species_n3",
    "species_richness_abundance_n_gram_species_n3",
    # Duplicates: dropped by the duplicate check on the Bose log (as done by Selin), HOWEVER can differ on other logs
    # because their values there are identical to a measure above.
    #"num_unique_sequences",  # same as num_sequences
    #"coverage (n=(2,))",  # same as coverage_n_gram_species_n2
    #"coverage (n=(3,))",  # same as coverage_n_gram_species_n3
    #"coverage_alternative (n=(2,))",  # same as coverage_alternative_n_gram_species_n2
    #"coverage_alternative (n=(3,))",  # same as coverage_alternative_n_gram_species_n3
    #"estimate_asymptotic_exp_shannon_entropy_abundance (n=(2,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n2
    #"estimate_asymptotic_exp_shannon_entropy_abundance (n=(3,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n3
    #"estimate_asymptotic_simpson_diversity_abundance (n=(2,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2
    #"estimate_asymptotic_simpson_diversity_abundance (n=(3,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3
    #"estimate_asymptotic_species_richness_abundance (n=(2,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n2
    #"estimate_asymptotic_species_richness_abundance (n=(3,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n3
    #"exp_shannon_entropy_abundance (n=(2,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n2
    #"exp_shannon_entropy_abundance (n=(3,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n3
    #"simpson_diversity_abundance (n=(2,))",  # same as simpson_diversity_abundance_n_gram_species_n2
    #"simpson_diversity_abundance (n=(3,))",  # same as simpson_diversity_abundance_n_gram_species_n3
    #"species_richness_abundance (n=(2,))",  # same as species_richness_abundance_n_gram_species_n2
    #"species_richness_abundance (n=(3,))",  # same as species_richness_abundance_n_gram_species_n3
]


# ─── Inputs of the run ────────────────────────────────────────────────────────
# Note: RUN_NAME is used as the output folder name, so it can only contain letters, digits, '_' and '-'.
# PEN, N_WINDOWS and MIN_CLUSTER_SIZE values are taken from the paper

FEATURE_MODE = "similarity"       # "similarity" | "combined" | "complexity"
MODE_NAME = {"similarity": "standard"}.get(FEATURE_MODE, FEATURE_MODE)
RUN_NAME = f"vc_{MODE_NAME}"

FUSION = "weighted_concat"      # "weighted_concat" | "concat" (only for "combined")
ALPHA = 0.5                     # weight of the complexity features in [0, 1] (only for "combined")
MEASURES = ALL_MEASURES # ["species_richness_abundance_n_gram_species_n3"]  # or ALL_MEASURES; not used for "similarity"
PEN = {"Bose": 3.0, "Ceravolo": 3.0, "Ostovar": 3.0}
N_WINDOWS = {"Bose": 75, "Ceravolo": 75, "Ostovar": 75}
MIN_CLUSTER_SIZE = {"Bose": 5, "Ceravolo": 5, "Ostovar": 5}
THRESHOLD = 0.3
LAG_WINDOW = 200
EVALUATE_NOISY = False
DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"


def detect_one(log, run_name, feature_mode, fusion, alpha, measures, pen, n_windows, min_cluster_size,
               threshold, datasets_path):
    """Runs the approach on one log; returns its row of detections.csv, or None if the log is skipped."""
    dataset = log["dataset"]
    log_name = log["log_filename"].split(".")[0]
    N = n_windows[dataset]
    try:
        actual_trace_cps = ra.load_ground_truth(log)
        if len(actual_trace_cps) == 0:
            return None
        # Prefer the plain .xes (e.g. the preprocessed Bose log of VC_execution.ipynb) over the .xes.gz
        log_dir = f"{datasets_path}/{dataset}"
        log_file = log["log_filename"]
        if not os.path.exists(f"{log_dir}/{log_file}"):
            log_file = f"{log_file}.gz"
        event_log = utils.import_event_log(path=log_dir, name=log_file)
        filtered_log = utils.filter_complete_events(event_log)
        Z, z_borders = None, None
        if feature_mode != "similarity":
            Z, z_borders = cca.load_complexity_matrix(datasets_path, dataset, log["log_filename"], measures, N)

        start_time = time.time()
        detected_windows, _, _, _, window_information, _ = cca.detect_log(
            filtered_log, N, min_cluster_size[dataset], threshold, pen[dataset], feature_mode, fusion, alpha,
            Z, z_borders)
        execution_time = time.time() - start_time

        # Window index -> trace id, exactly as the baseline (cluster_approach._run_processing_pipeline)
        detected_trace_cps = ca.get_changepoints_trace_idx(detected_windows, window_information, N)
    except Exception as e:
        print(f"Error processing {dataset}/{log_name}: {e!r}")
        return None
    # Same columns as algorithm_results_cdrift.csv, so the rows can be appended to it
    return {
        "Algorithm": run_name,
        "Log Source": dataset,
        "Log": log_name,  # without extension, as in algorithm_results_cdrift.csv
        "Detected Changepoints": detected_trace_cps,
        "Actual Changepoints for Log": actual_trace_cps,
        "Duration (Seconds)": execution_time,
        "Duration": time.strftime("%H:%M:%S", time.gmtime(execution_time)),
        "Window Size": N,
        "Threshold": threshold,
    }


def run(run_name, feature_mode="similarity", fusion="weighted_concat", alpha=0.5, measures=None,
        pen=ra.DEFAULT_PEN, n_windows=ra.DEFAULT_N_WINDOWS, min_cluster_size=None, threshold=0.5,
        lag_window=200, datasets_path="datasets/cdrift", out_root="baselines_results", evaluate_noisy=False):
    """Runs the clustering approach with the given parameters and evaluates it like run_approach.run().

    Args:
        measures: complexity measure names; None uses all measures left after dropping exact duplicates.
        pen, n_windows, min_cluster_size: dicts with the keys Bose, Ceravolo and Ostovar.
        min_cluster_size defaults to 10 for every dataset.
    """
    if min_cluster_size is None:
        min_cluster_size = {d: 10 for d in ra.DATASETS}
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_name):
        raise ValueError(f"run_name may only contain letters, digits, '_' and '-', got {run_name!r}")
    ra._check_dataset_dict("pen", pen)
    ra._check_dataset_dict("n_windows", n_windows)
    ra._check_dataset_dict("min_cluster_size", min_cluster_size)
    if feature_mode != "similarity":
        if measures is None:
            measures = ra.unique_measure_names(datasets_path, n_windows)
        valid = ra.all_measure_names(datasets_path, n_windows)
        unknown = [m for m in measures if m not in valid]
        if unknown or not measures:
            raise ValueError(f"Unknown or empty measures {unknown}. Valid measures:\n  " + "\n  ".join(valid))
    cca.check_params(feature_mode, fusion, alpha, measures)

    out_dir = f"{out_root}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    config = {"run_name": run_name, "approach": cca.algorithm_name(feature_mode, fusion),
              "feature_mode": feature_mode,
              "fusion": fusion if feature_mode == "combined" else None,
              "alpha": alpha if feature_mode == "combined" else None,
              "measures": list(measures) if feature_mode != "similarity" else None,
              "pen": pen, "n_windows": n_windows, "min_cluster_size": min_cluster_size, "threshold": threshold,
              "lag_window": lag_window, "evaluate_noisy": evaluate_noisy}
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Detection
    logs = ra.list_logs(datasets_path)
    if feature_mode != "similarity":
        ra.ensure_complexity_vectors(logs, n_windows, datasets_path)
    detect = partial(detect_one, run_name=run_name, feature_mode=feature_mode, fusion=fusion, alpha=alpha,
                     measures=measures, pen=pen, n_windows=n_windows, min_cluster_size=min_cluster_size,
                     threshold=threshold, datasets_path=datasets_path)
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
        pen=PEN,
        n_windows=N_WINDOWS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        threshold=THRESHOLD,
        lag_window=LAG_WINDOW,
        datasets_path=DATASETS_PATH,
        out_root=OUT_ROOT,
        evaluate_noisy=EVALUATE_NOISY,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
