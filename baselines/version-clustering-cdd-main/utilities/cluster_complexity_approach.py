"""Second version of the clustering-based concept drift detection approach where the input
feature of HDBSCAN can be chosen:

    "similarity": windowed DFG similarity vectors only (same as cluster_approach.py).
    "combined":   similarity vectors + complexity vectors, combined with one of the fusions:
                  "weighted_concat":  hstack[(1-alpha) * S / sqrt(N), alpha * Z / sqrt(k)]
                  "concat":           hstack[S, Z]
    "complexity": complexity vectors only.

S is the N x N similarity matrix, Z the N x k matrix of the selected complexity measures
(z-scored per log). The complexity vectors are loaded from datasets/cdrift/<dataset>[/nw<N>]
and computed (and cached) if they are missing, using run_approach.py from the repo root."""

# Standard library
import os
import sys
import json
import time
from functools import partial
from pathlib import Path

# Data manipulation
import numpy as np

# Custom utilities
import utilities.utilities as utils
import utilities.preprocessing_methods as ppm
import utilities.cluster_approach as ca

FEATURE_MODES = ["similarity", "combined", "complexity"]
FUSIONS = ["weighted_concat", "concat"]

# utilities -> version-clustering-cdd-main -> baselines -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS_PATH = str(REPO_ROOT / "datasets" / "cdrift")


def check_params(feature_mode, fusion, alpha, measures):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")
    if feature_mode == "combined" and fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if feature_mode != "similarity" and not measures:
        raise ValueError(f"feature_mode {feature_mode!r} needs a non-empty list of measures")


def algorithm_name(feature_mode, fusion):
    return f"Bernold_{feature_mode}_{fusion}" if feature_mode == "combined" else f"Bernold_{feature_mode}"


#############################################################################################################
########################################### COMPLEXITY VECTORS ##############################################
#############################################################################################################

def load_complexity_matrix(datasets_path, dataset, log_filename, measures, n_windows):
    """Loads the z-scored complexity matrix (n_windows x len(measures)) of one log, computing and
    caching the complexity vectors first if they don't exist for n_windows.

    Args:
        datasets_path: Folder with one subfolder per dataset, e.g. "<repo>/datasets/cdrift".
        dataset: Dataset subfolder, e.g. "Ceravolo".
        log_filename: Log file name without .gz, e.g. "bose_log.xes".

    Returns:
        Tuple[np.ndarray, list]: z-scored complexity matrix and the window borders it was computed with
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    import run_approach as ra

    ra.ensure_complexity_vectors([{"dataset": dataset, "log_filename": log_filename}],
                                 {dataset: n_windows}, datasets_path)
    log_name = log_filename.split(".")[0]
    folder = ra.find_vector_folder(datasets_path, dataset, log_name, n_windows)
    Z, _ = ra.load_standardized_matrix(folder, log_name, measures)
    with open(f"{folder}/{log_name}_borders.json") as f:
        z_borders = json.load(f)
    return Z, z_borders


#############################################################################################################
################################################ DETECTION ##################################################
#############################################################################################################

def cluster_features(similarity, Z, feature_mode, fusion, alpha, min_cluster_size, threshold):
    """Clusters the windows on the chosen input feature.

    Returns:
        Tuple[np.ndarray, np.ndarray]: cluster probabilities (input of the change point detection)
        and cluster labels per window
    """
    if feature_mode == "similarity":
        return ca.cluster_hdbscan(similarity, min_cluster_size=min_cluster_size, threshold=threshold)
    if feature_mode == "complexity":
        return ca.cluster_hdbscan(Z, min_cluster_size=min_cluster_size, threshold=threshold)

    if fusion == "weighted_concat":
        # Each block is divided by sqrt(number of columns), so the N similarity columns don't
        # outweigh the k complexity columns just because there are more of them.
        X = np.hstack([(1 - alpha) * similarity / np.sqrt(similarity.shape[1]),
                       alpha * Z / np.sqrt(Z.shape[1])])
        return ca.cluster_hdbscan(X, min_cluster_size=min_cluster_size, threshold=threshold)

    # concat
    X = np.hstack([similarity, Z])
    return ca.cluster_hdbscan(X, min_cluster_size=min_cluster_size, threshold=threshold)


def detect_log(filtered_log, n_windows, min_cluster_size, threshold, pen,
               feature_mode="similarity", fusion=None, alpha=0.5, Z=None, z_borders=None):
    """Steps 1-3 of the approach (features, clustering, change point detection) for one filtered log.

    Returns:
        Tuple: predicted change points (window indices), cluster labels, dfg, borders,
        window information and date information of the log
    """
    # Step 1: Measuring Similarity Across Windows
    dfg, borders, window_information, log_date_info = ppm.log_to_windowed_dfg_count(filtered_log, n_windows)
    similarity = None
    if feature_mode != "complexity":
        similarity = ppm.similarity_calculation(np.array(dfg))
        if similarity.shape[0] == 0:
            raise ValueError("Empty dataset passed to cluster_hdbscan")
    if Z is not None:
        assert [list(b) for b in borders] == [list(b) for b in z_borders], \
            "Windows of the complexity vectors don't match the DFG windows"

    # Step 2: Clustering Windows into Process Versions
    probabilities, clusters_labels = cluster_features(similarity, Z, feature_mode, fusion, alpha,
                                                      min_cluster_size, threshold)

    # Step 3: Change Point Detection
    pred_drift = ca.find_transition_steps(probabilities, pen, n_windows)

    return pred_drift, clusters_labels, dfg, borders, window_information, log_date_info


#############################################################################################################
############################################### PIPELINE ####################################################
#############################################################################################################

def processing_pipeline_general_complexity(log_dir, n_windows=100, min_cluster_size=10, threshold=0.5, pen=1,
                                           feature_mode="similarity", fusion="weighted_concat", alpha=0.5,
                                           measures=None, datasets_path=DEFAULT_DATASETS_PATH):
    """Same as cluster_approach.processing_pipeline_general, with the input feature chosen by
    feature_mode / fusion / alpha. The complexity vectors of the logs in log_dir are taken from
    datasets_path/<name of log_dir>.

    Args:
        log_dir: Path of the directory where all the data is stored.
        n_windows (int, optional): Number of windows. Defaults to 100.
        measures (list): Complexity measures to use (see run_experiments.ALL_MEASURES).
    """
    check_params(feature_mode, fusion, alpha, measures)
    dataset_name = log_dir.split("/")[-1]

    drift_info = None
    if os.path.exists(f"{log_dir}/drift_info.csv"):
        drift_info = ca.load_drift_info(f"{log_dir}/drift_info.csv")
        drift_info["time_seconds"] = 0 # Adds an execution time column

    process_fn = partial(process_log_complexity, feature_mode=feature_mode, fusion=fusion, alpha=alpha,
                         measures=measures, datasets_path=datasets_path)
    return ca._run_processing_pipeline(log_dir, drift_info, n_windows, min_cluster_size, threshold, pen,
                                       track_algorithm_results=True, dataset_name=dataset_name,
                                       process_fn=process_fn,
                                       algorithm_name=algorithm_name(feature_mode, fusion))


def process_log_complexity(name_path, drift_info, n_windows, min_cluster_size, threshold, pen,
                           feature_mode, fusion, alpha, measures, datasets_path):
    """Same as cluster_approach.process_log, with the input feature chosen by feature_mode."""
    try:
        start_time = time.time()       # Record start

        name, path = name_path

        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        number_of_traces = utils.get_number_of_traces(filtered_log)

        Z, z_borders = None, None
        if feature_mode != "similarity":
            dataset = os.path.basename(os.path.normpath(path))
            Z, z_borders = load_complexity_matrix(datasets_path, dataset, name, measures, n_windows)

        # Steps 1-3
        pred_drift, clusters_labels, dfg, borders, window_information, log_date_info = detect_log(
            filtered_log, n_windows, min_cluster_size, threshold, pen, feature_mode, fusion, alpha, Z, z_borders)

        # Step 4: create output files

        # Get real drift points
        real_drift = []
        if drift_info is not None:
            real_drift = ca.process_drift(drift_info, name, borders)

        # Save results

        name_no_extention = name.split(".")[0] # removes the .xes
        np.savetxt(f"{path}/outputs/{name_no_extention}.txt", pred_drift, fmt='%d')
        filename_hdbscan = f"{path}/outputs/{name_no_extention}_output.svg"
        ca.save_plot_drift(clusters_labels,real_drift,pred_drift,filename_hdbscan)

        # Create output for visualization framework
        output_one, output_two = ca.to_vis_system(dfg,n_windows,clusters_labels,pred_drift,window_information)
        output_one.to_csv(f"{path}/outputs/{name_no_extention}_general.csv",index=False)
        output_two.to_csv(f"{path}/outputs/{name_no_extention}_df_info.csv",index=False)

        end_time = time.time()         # Record end
        execution_time = end_time - start_time

        return name, pred_drift, real_drift, execution_time, borders, window_information, log_date_info, number_of_traces
    except Exception as e:
        # In case of an error, the log is skipped. Printed so that e.g. a wrong measure name doesn't go unnoticed.
        print(f"Error processing {name_path[0]}: {e!r}")
        return None, None, None, None, None, None, None, None
