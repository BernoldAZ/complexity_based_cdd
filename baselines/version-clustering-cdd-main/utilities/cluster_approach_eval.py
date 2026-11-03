"""Evaluation pipeline for the clustering-based drift detection approach: reads
predicted and ground-truth change points per log, computes precision/recall/F1/lag
per relative lag factor (in parallel across logs), and saves per-class classification
report plots and summary text files."""

###########################################################################################################################
# Collection of functions for evaluation

import pandas as pd
from typing import List, Tuple, Union
import os
import json
from math import inf
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

from . import cdrift_metrics_2d as cdrift
from . import utilities as utils
from . import config as cfg

###########################################################################################################################
# Main evaluation pipeline function
def evaluation_pipeline(test_dir, output_dir, n_windows):
    drift_info_path = f"{test_dir}/Annotations"
    window_info_path = f"{test_dir}/Annotations"
    traces_path = f"{test_dir}/Annotations"
    category_index = get_ex_decoder()
    
    eval_results = defaultdict(lambda: defaultdict(dict))
    utils.make_dir(output_dir)
    
    folder_path = f"{test_dir}/outputs"
    all_predictions = read_pred_files(folder_path)
    
    #for pred in all_predictions:
     #   results_for_lags = process_prediction(pred,drift_info=drift_info, window_info=window_info,category_index=category_index, traces_per_log=traces_per_log)

    # Parallelize the second loop
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    # Create the pool with preloaded data
    with multiprocessing.Pool(
        num_cores,
        initializer=init_worker,
        initargs=(drift_info_path, window_info_path, traces_path, category_index,n_windows)
    ) as pool:
        results_list = list(tqdm(
            pool.imap_unordered(process_prediction, all_predictions),
            total=len(all_predictions),
            desc="Evaluating predictions", unit=" log"
        ))

    # Aggregate results
    for result_dict in results_list:
        for lag_factor, result_data in result_dict.items():
            log_xes = result_data["log_xes"]
            eval_results[f"lag_{lag_factor}"][log_xes] = result_data["data"]

    eval_results = dict(eval_results)
    traces_per_log = get_traces_per_log(traces_path)
    for lag_factor in cfg.RELATIVE_LAG:
        str_lag = f"lag_{lag_factor}"
        results_df = save_results(eval_results[str_lag], lag_factor, output_dir)
        print_measures(results_df, traces_per_log, lag_factor, output_dir)
        plot_classification_report(results_df, output_dir, lag_factor)


_shared = {}
def init_worker(drift_info_path, window_info_path, traces_path, category_index,n_windows):
    """Initialize each worker process with heavy shared data."""

    global _shared
    _shared['drift_info'] = get_drift_info_updated_windows(drift_info_path)
    _shared['window_info'] = get_window_info(window_info_path)
    _shared['traces_per_log'] = get_traces_per_log(traces_path)
    _shared['category_index'] = category_index
    _shared['n_windows'] = n_windows

###########################################################################################################################



def read_pred_files(folder_path):
    """
    Reads all .txt files from a folder and stores their contents as a list of lists.
    Each file's numbers (one per line) are converted to integers.
    Empty files produce an empty list.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing .txt files.
    
    Returns
    -------
    list_of_lists : list of lists
        Each sublist contains the integers from one file.
    """
    list_of_lists = []
    
    # List all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Iterate over each file
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # Convert non-empty lines to integers, strip newline characters
                numbers = [int(line.strip())+1 for line in lines if line.strip()]
                list_of_lists.append((numbers,filename))
        except Exception as e:
            # In case of any read error, append empty list
            print(f"Warning: Could not read {filename}: {e}")
            list_of_lists.append(([],filename))
    
    return list_of_lists

###########################################################################################################################
# Collection of functions for evaluation

def process_prediction(pred_log):
    """Helper function to process one prediction tuple in parallel."""
    """Process one prediction tuple using shared data."""
    global _shared

    drift_info = _shared['drift_info']
    window_info = _shared['window_info']
    category_index = _shared['category_index']
    traces_per_log = _shared['traces_per_log']
    n_windows = _shared['n_windows']

    pred_windows, log_file = pred_log

    

    # Get log info
    log_xes = log_file.split(".")[0] + ".xes"
    log_info = get_log_info(log_xes, drift_info)
    log_window_info = window_info[log_xes]


    y_pred_category = ["sudden"] * len(pred_windows)
    pred_change_points = get_changepoints_trace_idx(pred_windows, y_pred_category, log_window_info, n_windows)

    # Real windows
    true_change_points, y_true_category = get_true_changepoints_and_classes(log_info)
    #true_change_points, y_true_category = change_all_sudden(true_change_points, y_true_category)

    results_for_lags = {}

    for lag_factor in cfg.RELATIVE_LAG:
        metrics, y_pred_sorted, y_true_sorted = get_evaluation_metrics(
            y_true=true_change_points,
            y_pred=pred_change_points,
            y_true_label=y_true_category,
            y_pred_label=y_pred_category,
            factor=lag_factor,
            number_of_traces=traces_per_log[log_xes])

        results_for_lags[lag_factor] = {
            "log_xes": log_xes,
            "data": {
                "Detected Changepoints": pred_change_points,
                "Actual Changepoints": true_change_points,
                "Predicted Drift Types": y_pred_category,
                "Actual Drift Types": y_true_category,
                "F1-Score": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Average Lag": metrics["lag"],
                "y_pred_sorted": y_pred_sorted,
                "y_true_sorted": y_true_sorted,
                "n_traces": traces_per_log[log_xes]
            }
        }

    return results_for_lags

def change_all_sudden(true_change_points, y_true_category):
    unique_change_points = list({(x,x) for pair in true_change_points for x in pair})
    all_sudden = ["change_point"] * len(unique_change_points)
    return unique_change_points, all_sudden

    
##############################################################################################################
##############################################################################################################
# MODIFY TO CHANGE HOW THE NN OUTPUT LAYER WORKS.
# CURRENTLY IT TAKES EVERYTHING AS A SUDDEN DRIFT.
##############################################################################################################
##############################################################################################################
def get_changepoints_trace_idx(window_idx: list, y_pred: list, window_info: dict, n_windows : int) -> List[tuple]:
    """Get changepoints as trace indices for WINSIM encoding. 

    Args:
        window_idx (list): List of window indices where a drift happened
        y_pred (list): List of predictions
        window_info (dict): Window info

    Returns:
        List[tuple]: List of changepoints as trace indices
    """

    change_points = []
    if len(window_idx) == 0:
        return change_points
    else:
        # window_idx and y_pred have same size, i works for both
        #print(window_idx,y_pred)
        for i in range(len(window_idx)): 
            if y_pred[i] == "sudden": # if y_pred[i] == "change_point":
                # changepoint is equal to the date of the first trace in the middle
                # window of bbox
                change_point = window_idx[i]
                if change_point > n_windows:
                    change_point = n_windows
                elif change_point < 1:
                    change_point = 1
                change_point_trace_id = (int(window_info[str(change_point)][0]),
                                         int(window_info[str(change_point)][0]))
            #else:
                # TODO: 
                # 
                
            change_points.append(change_point_trace_id)
        return change_points

def match_labels(assignments: list, y_true: list, y_true_labels: list,
                 y_pred: list, y_pred_labels: list) -> list:
    """Match predicted label with actual label. 
    If predicted label is not equal to actual label, 
    remove prediction from assignments.
    In other words: prediction becomes a false positive.

    Args:
        assignments (list): Assignments from CDrift evaluation
        y_true (list): Actual changepoints
        y_true_labels (list): Actual labels
        y_pred (list): Predicted changepoints
        y_pred_labels (list): Predicted labels

    Returns:
        list: Label-matched assignments
    """
    matched_assignments = []
    for assignment in assignments:
        pred, true = assignment
        if y_pred_labels[y_pred.index(pred)] == y_true_labels[y_true.index(true)]:
            matched_assignments.append(assignment)
    return matched_assignments

def get_evaluation_metrics(y_true: list, y_pred: list,
                           y_true_label: list, y_pred_label: list,
                           factor: float, number_of_traces: int) \
        -> Tuple[dict, list, list]:
    """Get all relevant evaluation metrics.

    Args:
        y_true (list): List of groundtruth values
        y_pred (list): List of prediction values
        y_true_label (list): List of groundtruth labels
        y_pred_label (list): List of predicted labels
        factor (float): Factor for relative lag
        number_of_traces (int): Number of traces for event log

    Returns:
        Tuple[dict,list,list]: Dict, containing evaluation measures. 
        And sorted lists containing predicted and groundtruth labels
    """
    lag = int(factor * number_of_traces)

    _, assignments = cdrift.getTP_FP(detected=y_pred,
                                     known=y_true,
                                     lag=lag)

    matched_assignments = match_labels(assignments, y_true, y_true_label,
                                       y_pred, y_pred_label)

    if len(matched_assignments) > 0:
        preds, trues = zip(*matched_assignments)
    else:
        preds = trues = []

    sorted_trues = sorted(
        y_true, key=lambda x: trues.index(x) if x in trues else inf)
    
    y_pred_fp = [x for x in y_pred if x not in preds]
    y_pred_fp = [y_pred_label[y_pred.index(x)] for x in y_pred_fp]

    y_pred_label_sorted = [y_pred_label[y_pred.index(x)] for x in preds]
    y_true_label_sorted = [y_true_label[y_true.index(x)] for x in sorted_trues]

    # add FN
    while len(y_pred_label_sorted) < len(y_true_label_sorted):
        y_pred_label_sorted.append(np.NaN)
    
    # add FP    
    for elem in y_pred_fp:
        y_true_label_sorted.append(np.NaN)
        y_pred_label_sorted.append(elem)

    tp = len(matched_assignments)
    fp = len(y_pred) - tp

    f1, precision, recall = get_f1_score(tp, fp, len(y_true))
    average_lag = get_average_lag(matched_assignments)

    metrics = {"f1": f1,
               "precision": precision,
               "recall": recall,
               "lag": average_lag}
    return metrics, y_pred_label_sorted, y_true_label_sorted


def get_predicted_classes(y_pred: list, category_index: dict) -> list:
    """Get predicted drift types.

    Args:
        y_pred (list): List of prediction values
        category_index (dict): Mapping of drift types to drift IDs

    Returns:
        list: List of predicted drift types
    """
    y_pred_category = []
    for y in y_pred:
        # Convert from numpy array -> scalar int
        if isinstance(y, np.ndarray):
            y = int(y.item())   # or y = int(np.argmax(y)) if using probabilities
        y_pred_category.append(category_index.get(y, {}).get("name"))
    return y_pred_category

def get_log_info(log_name: str, drift_info: pd.DataFrame) -> pd.DataFrame:
    """Get drift info for event log.

    Args:
        log_name (str): Name of event log
        drift_info (pd.DataFrame): Drift info

    Returns:
        pd.DataFrame: DataFrame, containing drift info for event log
    """
    return drift_info.loc[drift_info["log_name"] == log_name]

def get_window_info(data_dir: str) -> Union[list, dict]:
    """Load window info from file.

    Args:
        data_dir (str): Directory where window info is stored

    Returns:
        Union[list, dict]: Window info
    """
    window_info_path = os.path.join(data_dir, "window_info.json")
    assert os.path.isfile(window_info_path), "No window info file found"
    file = open_file(window_info_path)
    return json.load(file)

def get_traces_per_log(data_dir: str) -> Union[list, dict]:
    """Load traces per log from file.


    Args:
        data_dir (str): Directory where traces per log info is stored

    Returns:
        Union[list, dict]: Number of traces per log
    """
    number_of_traces_path = os.path.join(data_dir, "number_of_traces.json")
    assert os.path.isfile(
        number_of_traces_path), "No number of traces file found"
    file = open_file(number_of_traces_path)
    return json.load(file)

def get_drift_info_updated_windows(dir: str) -> pd.DataFrame:
    """Get drift information.

    Args:
        dir (str): Directory, where drift information file is stored.

    Returns:
        pd.DataFrame: Drift information
    """
    path = os.path.join(dir, "drift_info_updated_windows.csv")
    return pd.read_csv(path, sep=",")

def get_true_changepoints_and_classes(log_info: pd.DataFrame) -> tuple:
    """Get true changepoints as trace indices and true drift types.

    Args:
        log_info (pd.DataFrame): Drift info for event log

    Returns:
        tuple: A tuple containing two lists:
            - List of true changepoints as trace indices.
            - List of true drift types.
    """
    drift_ids = pd.unique(log_info["drift_or_noise_id"])
    change_points_trace_idx = []
    drift_types = []

    for drift_id in drift_ids:
        drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]

        # Extract changepoints
        if len(drift.index) > 1:
            first_row = drift.iloc[0]["drift_traces_index"]
            first_row = utils.special_string_2_list(first_row)
            last_row = drift.iloc[-1]["drift_traces_index"]
            last_row = utils.special_string_2_list(last_row)
            change_points_trace_idx.append((first_row[0], last_row[-1]))

            # Extract drift type
            drift_type = drift.iloc[0]["drift_type"]
            drift_types.append(drift_type)

            # Loop through subdrifts using iterrows()
            for _, subdrift in drift.iterrows():
                first_row = subdrift["drift_traces_index"]
                first_row = utils.special_string_2_list(first_row)
                last_row = subdrift["drift_traces_index"]
                last_row = utils.special_string_2_list(last_row)
                change_points_trace_idx.append((first_row[0], last_row[-1]))

                # Extract change type for subdrift
                drift_type = subdrift["change_type"]
                drift_types.append(drift_type)

        else:
            trace_id = drift.iloc[0]["drift_traces_index"]
            trace_id = utils.special_string_2_list(trace_id)
            change_points_trace_idx.append((trace_id[0], trace_id[-1]))

            # Extract drift type
            drift_type = drift.iloc[0]["drift_type"]
            drift_types.append(drift_type)

    return change_points_trace_idx, drift_types

def open_file(path: str):
    """Opens file from path.

    Args:
        path (str): Filepath

    Returns:
        TextIOWrapper: Opened file
    """
    return open(path)

def get_ex_decoder() -> dict:
    """Get decoders for classes.

    Returns:
        dict: Category index for drift types 
    """
    category_index = {
        1: {
            "id": 1,
            "name": "sudden"
        },
        2: {
            "id": 2,
            "name": "gradual"
        },
        3: {
            "id": 3,
            "name": "incremental"
        },
        4: {
            "id": 4,
            "name": "recurring"
        }
    }

    return category_index

def get_precision(tp: int, fp: int) -> float:
    """Get precision.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives

    Returns:
        float: Precision
    """
    return tp / (tp+fp)


def get_recall(tp: int, y_true_size: int) -> float:
    """Get recall.

    Args:
        tp (int): Number of true positives
        y_true_size (int): Size of true values

    Returns:
        float: Recall
    """
    return tp / y_true_size


def get_f1_score(tp: int, fp: int, y_true_size: int) -> Tuple[float, float, float]:
    """Get F1-score.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        y_true_size (int): Size of true values

    Returns:
        Union[Tuple[float,float,float], Tuple[np.nan, np.nan, np.nan]]: F1-score
    """
    try:
        precision = get_precision(tp, fp)
        recall = get_recall(tp, y_true_size)

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score, precision, recall
    except ZeroDivisionError:
        return np.nan, np.nan, np.nan

def get_average_lag(assignments: List[Tuple[int, int]]):
    """Source: 
    https://github.com/cpitsch/cdrift-evaluation/blob/main/cdrift/evaluation.py
    Author: cpitsch
    Note: The code was adapted so that it also works with two-dimensional tuples or 
    change points. Therefore, the description of the function has also been changed.
    Calculates the average lag between detected and actual start and end changepoints 
    (Caution: false positives do not affect this metric!)

    Args:
        assignments (List[Tuple[int,int]]): List of actual and detected changepoint 
            assignments

    Returns:
        float: the average distance between detected changepoints and the actual 
            changepoint they get assigned to
    """
    avg_lag = 0
    for (dc, ap) in assignments:
        avg_lag += abs(dc[0] - ap[0])
        avg_lag += abs(dc[1] - ap[1])
    try:
        return avg_lag / len(assignments)
    except ZeroDivisionError:
        return np.nan

def get_f1_score_per_class(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Get F1-score per class.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        Union[Tuple[float,float,float], Tuple[np.nan, np.nan, np.nan]]: F1-score
    """
    try:
        precision = get_precision(tp, fp)
        recall = get_recall(tp, (tp+fn))

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score, precision, recall
    except ZeroDivisionError:
        return 0, 0, 0

def get_FP_TP_per_class(y_pred: list, y_true: list) -> dict:
    """Get true positives, false positives and false negatives per class.

    Args:
        y_pred (list): Predictions (classes)
        y_true (list): Groundtruth (classes)

    Returns:
        dict: Dictionary containing TP, FN and FP per class
    """
    measures_per_class = defaultdict(lambda: defaultdict(int))
    for i, true in enumerate(y_true):
        if true == y_pred[i]:
            measures_per_class[true]["TP"] += 1
        elif pd.isna(y_pred[i]):
            measures_per_class[true]["FN"] += 1
        elif pd.isna(true):
            measures_per_class[y_pred[i]]["FP"] += 1
    return dict(measures_per_class)

# Outputs #####################################################

def save_results(results: dict, lag_factor: float, val_path: str) -> pd.DataFrame:
    """Saves evaluation results to output path.

    Args:
        results (dict): Evaluation measures
        lag_factor (float): Relative lag
        val_path (str): Evaluation directory
    
    Returns:
        pd.DataFrame: DataFrame, containing evaluation measures
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(val_path,
                             f"evaluation_results_general_{lag_factor}_lag.csv")
    results_df.to_csv(save_path, sep=",")
    return results_df

def print_measures(results: pd.DataFrame, num_traces: dict,
                   lag_factor: float, val_path: str):
    """Generate evaluation measure overview. 
    Calculates total average F1-score and total average lag.

    Args:
        results (pd.DataFrame): Evaluation results
        num_traces (dict): Number of traces per log
        lag_factor (float): Relative lag
        val_path (str): Evaluation directory
    """
    num_events = len(num_traces.keys())

    f1_values = np.nansum(results["F1-Score"])
    precision_values = np.nansum(results["Precision"])
    recall_values = np.nansum(results["Recall"])
    lag_values = np.nansum(results["Average Lag"])

    total_average_f1 = f1_values / num_events
    total_average_precision = precision_values / num_events
    total_average_recall = recall_values / num_events
    total_average_lag = lag_values / num_events
    average_length = int(sum(num_traces.values()) / num_events)

    text_path = os.path.join(val_path,
                             f"evaluation_report_general_{lag_factor}_lag.txt")

    with open(text_path, "w") as f:
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write("EVALUATION REPORT:")
        f.write("\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write(f"In total {num_events} were evaluated, with an average trace length of\
                {average_length}\n")
        f.write("\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average f1 score for all evaluated logs is: {total_average_f1}.\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average precision score for all evaluated logs is: {total_average_precision}.\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average recall score for all evaluated logs is: {total_average_recall}.\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average lag for all evaluated logs is: {total_average_lag}.\n")
        f.write(
            "---------------------------------------------------------------------\n")

def plot_classification_report(results: pd.DataFrame, path: str, lag_factor: float):
    """Plot classification report.

    Args:
        results (pd.DataFrame): Evaluation results
        path (str): Save path
        lag_factor (float): Lag factor used
    """
    y_trues = results["y_true_sorted"].to_numpy()
    y_preds = results["y_pred_sorted"].to_numpy()

    y_trues_flat = [elem for sub in y_trues for elem in sub]
    y_preds_flat = [elem for sub in y_preds for elem in sub]

        
    results = get_FP_TP_per_class(y_preds_flat, y_trues_flat)
    
    c_r = {}
    
    average_precision = 0
    average_recall = 0
    average_f1 = 0
    
    for k in results.keys():
        f1, precision, recall = get_f1_score_per_class(tp=results[k]["TP"], 
                                                       fp=results[k]["FP"], 
                                                       fn=results[k]["FN"])
        c_r[k] = {"precision": precision, 
                  "recall": recall, 
                  "f1-score": f1}
        average_precision += precision
        average_recall += recall
        average_f1 += f1
        
    # sort dict to align output    
    c_r = dict(sorted(c_r.items()))
        
    average_precision = average_precision / len(c_r)
    average_recall = average_recall / len(c_r)
    average_f1 = average_f1 / len(c_r)   
    c_r["average"] = {"precision": average_precision, 
                      "recall": average_recall, 
                      "f1-score": average_f1}

    # exclude support from clf report with iloc
    clf_r = sns.heatmap(pd.DataFrame(
        c_r).T, cmap="YlGn", annot=True, cbar=False)
    clf_r_fig = clf_r.get_figure()
    save_path = os.path.join(
        path, f"classification_report_general_{lag_factor}_lag.png")
    clf_r_fig.tight_layout()
    clf_r_fig.savefig(save_path, dpi=300)
    print(f"Classification report is saved at: {save_path}")
    plt.close(clf_r_fig)