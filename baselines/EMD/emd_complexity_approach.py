"""Earth Mover's Distance (earthmover.py) with complexity measures, where the signal that the peaks are
detected on can be chosen. For every EMD position i the two adjacent windows w1 = traces[i:i+w] and
w2 = traces[i+w:i+2w] are compared:

    "standard":   d_emd, the EMD between the stochastic languages of w1 and w2 (same as earthmover.py).
    "combined":   d_emd and d_cx combined with one of the fusions:
                  "weighted_concat":  sqrt(((1-alpha) * d_emd)^2 + (alpha * d_cx / sqrt(k))^2)
                  "concat":           sqrt(d_emd^2 + d_cx^2)
    "complexity": d_cx only.

d_cx = ||z(w1) - z(w2)||, with z the k selected complexity measures of a window (z-scored per log over all
windows). Since ||hstack[a*X, b*Z]|| = sqrt(a^2 * d_X^2 + b^2 * d_Z^2), the fusions are the window-pair
counterpart of the concatenations in cluster_complexity_approach.py.

The complexity vectors are computed exactly on every EMD window and cached in
datasets/cdrift/<dataset>/emd_w<w>_s<s>/."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from pm4py.objects.log.obj import EventLog

import earthmover as emd

FEATURE_MODES = ["standard", "combined", "complexity"]
FUSIONS = ["weighted_concat", "concat"]

# EMD -> baselines -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def check_params(feature_mode, fusion, alpha, measures):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")
    if feature_mode == "combined" and fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if feature_mode != "standard" and not measures:
        raise ValueError(f"feature_mode {feature_mode!r} needs a non-empty list of measures")


def algorithm_name(feature_mode, fusion):
    return f"EMD_{feature_mode}_{fusion}" if feature_mode == "combined" else f"EMD_{feature_mode}"


#############################################################################################################
########################################### COMPLEXITY VECTORS ##############################################
#############################################################################################################

def window_starts(n_traces, window_size, step_size):
    """First trace of every window compared by calculateDistSeriesStride: w1 starts at i, w2 at i + w."""
    positions = range(0, n_traces - 2 * window_size, step_size)
    return sorted({i for i in positions} | {i + window_size for i in positions})


def window_complexity_vectors(event_log, window_size, step_size, cache_folder, log_name):
    """Complexity vectors (all measures, not standardized) of every EMD window of one log, computed and
    cached in cache_folder if they don't exist yet.

    Returns:
        Tuple[np.ndarray, list, list]: one vector per window, the measure names and the window starts
    """
    paths = {name: f"{cache_folder}/{log_name}_{name}"
             for name in ["complexity_vectors.npy", "measure_names.json", "window_starts.json"]}
    if all(os.path.exists(path) for path in paths.values()):
        with open(paths["measure_names.json"]) as f:
            measure_names = json.load(f)
        with open(paths["window_starts.json"]) as f:
            starts = json.load(f)
        return np.load(paths["complexity_vectors.npy"]), measure_names, starts

    sys.path.insert(0, str(REPO_ROOT / "complexity_measures"))
    from compute_log_complexity import compute_complexity_vector
    from windowing import filter_complete_events

    starts = window_starts(len(event_log), window_size, step_size)
    vectors, measure_names = [], None
    for j in starts:
        # Same traces as the EMD window; complete events only, as for the cached tumbling-window vectors
        window_log = EventLog(list(event_log)[j:j + window_size], attributes=event_log.attributes,
                              extensions=event_log.extensions, classifiers=event_log.classifiers)
        vector, measure_names = compute_complexity_vector(filter_complete_events(window_log))
        vectors.append(vector)
    vectors = np.array(vectors, dtype=float)

    os.makedirs(cache_folder, exist_ok=True)
    np.save(paths["complexity_vectors.npy"], vectors)
    with open(paths["measure_names.json"], "w") as f:
        json.dump(measure_names, f)
    with open(paths["window_starts.json"], "w") as f:
        json.dump(starts, f)
    return vectors, measure_names, starts


def complexity_distance_series(vectors, measure_names, starts, measures, n_traces, window_size, step_size):
    """d_cx = ||z(w1) - z(w2)|| at the same positions as calculateDistSeriesStride. The selected measures are
    z-scored per log over all windows (as run_approach.load_standardized_matrix)."""
    matrix = vectors[:, [measure_names.index(name) for name in measures]]
    stds = matrix.std(axis=0)
    Z = (matrix - matrix.mean(axis=0)) / np.where(stds < 1e-10, 1, stds)
    row = {start: idx for idx, start in enumerate(starts)}
    return np.array([np.linalg.norm(Z[row[i]] - Z[row[i + window_size]])
                     for i in range(0, n_traces - 2 * window_size, step_size)])


#############################################################################################################
################################################ DETECTION ##################################################
#############################################################################################################

def fuse(d_emd, d_cx, feature_mode, fusion, alpha, k):
    if feature_mode == "standard":
        return d_emd
    if feature_mode == "complexity":
        return d_cx
    if fusion == "weighted_concat":
        # d_cx is divided by sqrt(k), so the k measures don't outweigh d_emd (in [0, 1]) just because there are many
        return np.sqrt(((1 - alpha) * d_emd) ** 2 + (alpha * d_cx / np.sqrt(k)) ** 2)
    # concat
    return np.sqrt(d_emd ** 2 + d_cx ** 2)


def detect_change(event_log, window_size, step_size=2, feature_mode="standard", fusion=None, alpha=0.5,
                  measures=None, complexity=None):
    """EMD change point detection on the chosen signal; for "standard" the same as earthmover.detect_change.

    Args:
        complexity: (vectors, measure_names, starts) of window_complexity_vectors; not used for "standard".

    Returns:
        List[int]: detected change points (trace indices)
    """
    traces = emd.extractTraces(event_log)
    d_emd, d_cx = None, None
    if feature_mode != "complexity":
        d_emd = np.array(emd.calculateDistSeriesStride(traces, window_size, step_size, show_progressBar=False))
    if feature_mode != "standard":
        d_cx = complexity_distance_series(*complexity, measures, len(traces), window_size, step_size)
    signal = fuse(d_emd, d_cx, feature_mode, fusion, alpha, len(measures) if measures else 0)
    return emd.visualInspection_Stride(signal, window_size, step_size)
