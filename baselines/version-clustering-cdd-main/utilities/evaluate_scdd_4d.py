"""Evaluation pipeline for SCDD-style detection results at drift-type and
change-point granularity: loads detection/drift-info CSVs, matches detected drifts
to actual ones (per drift type and lag), computes TP/FP/precision/recall/F1 in
parallel across logs, and aggregates/saves the results (incl. LaTeX tables)."""

import ast
from utilities.cdrift_metrics_1d import getTP_FP, assign_changepoints
import numpy as np
import sys
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Optional, Any, Dict, List, Tuple
from utilities import config as cfg

RELATIVE_LAG = cfg.RELATIVE_LAG

def _weighted_f1_score(group):
    fn_tp_weighted_f1 = (group['F1_score'] * group['FN_TP']).sum() / group['FN_TP'].sum()
    return pd.Series({'Weighted_F1': fn_tp_weighted_f1})



def _get_weighted_F1_accuracy(report):
    # Group by 'noise_level', 'lag', and any other columns that are constant for each group
    grouped = report.groupby(['noise_level', "drift_type"])

    # Apply the custom function to calculate the weighted F1 score for each group
    weighted_F1_per_set = grouped.apply(_weighted_f1_score).reset_index()

    return weighted_F1_per_set


def get_total_evaluation_results(evaluation_report, grouping):
    aggregations = {
        'TP': 'sum',
        'FP': 'sum',
        'FN_TP': 'sum'}
    # grouping = ['noise_level', 'complexity', 'method', 'window_size', 'lag', 'eval_focus']
    #grouping = ['noise_level', 'complexity', 'n_drifts', "n_change_points", 'drift_type']
    #pdb.set_trace()

    # Convert Series objects to strings
    evaluation_report['noise_level'] = evaluation_report['noise_level'].astype(str)
    evaluation_report['drift_type'] = evaluation_report['drift_type'].astype(str)

    evaluation_report_agg = evaluation_report.groupby(grouping).agg(aggregations)

    evaluation_report_agg = evaluation_report_agg.assign(Precision=lambda x: get_precision(x['TP'], x['FP']))
    evaluation_report_agg = evaluation_report_agg.assign(Recall=lambda x: get_recall(x['TP'], x['FN_TP']))
    evaluation_report_agg = evaluation_report_agg.assign(F1_score=lambda x: get_f1_score(x['Precision'], x['Recall']))

    return evaluation_report_agg


def transform_tuples(original_list):
    """
    Transforms a list of tuples containing start and end values along with a drift type
    into a new list of tuples where each start value is paired with 'drift_type_start'
    and each end value is paired with 'drift_type_end'.

    Args:
    original_list (list): A list of tuples in the format [((start1, end1), drift_type1), ((start2, end2), drift_type2), ...]

    Returns:
    list: A new list of tuples in the format [(start1, 'drift_type1_start'), (end1, 'drift_type1_end'), ...]
    """
    new_list = []

    # Iterate over the original list of tuples
    for item in original_list:
        # Extract start and end values from the tuple
        start, end = item[0]
        drift_type = item[1]
        if item[1] == "gradual":
            # Create new tuples with gradual start and end infos
            new_list.append((start, drift_type + '_start'))
            new_list.append((end, drift_type + '_end'))
        else:
            # Create new tuples with sudden infos
            new_list.append((start, drift_type))

    return new_list

def add_no_drift_labels(drift_info: list):

    if len(drift_info) == 0:
        drift_info.append([0, "no_drift"])

    return drift_info


def get_precision(TP, FP):
    precision = np.where(TP + FP > 0, np.divide(TP, TP + FP), 0)
    return precision


def get_recall(TP, FN_TP):
    recall = np.where(FN_TP > 0, np.divide(TP, FN_TP), 0)
    return recall


def get_f1_score(precision, recall):

    f1_score = np.where(precision + recall > 0,  (2 * precision * recall) / (precision + recall), 0)

    return f1_score



def load_initial_drift_info_file(data_dir: str) -> pd.DataFrame:
    """Loads drift info from file.

    Args:
        data_dir (str): Directory where drift info is stored

    Returns:
        pd.DataFrame: Drift info
    """
    drift_info_path = os.path.join(data_dir, "drift_info.csv")
    print(drift_info_path)
    assert os.path.isfile(drift_info_path), "No drift info file found"
    return pd.read_csv(drift_info_path, sep=";")


def get_noise_info(log_name: str, noise_info: pd.DataFrame) -> pd.DataFrame:
    """Get noise info for event log.

    Args:
        log_name (str): Name of event log
        noise_info (pd.DataFrame): Drift info (filtered)

    Returns:
        pd.DataFrame: DataFrame, containing drift info for event log
    """
    #pdb.set_trace()
    value = str(0.0)
    if not log_name.endswith(".xes"):
        log_name = log_name + ".xes"
    if log_name in noise_info["log_name"].tolist():
        value = noise_info.loc[noise_info["log_name"] == log_name]["value"].values[0]

    return value


def main_cp_only_external(full_path, LAG):

    # Load csv file with evaluation info
    df = pd.read_csv(full_path)

    # Add actual drift info depending on the dataset from path:
    if os.path.basename(os.path.split(full_path)[0]) == "ostovar":
        df['Actual Changepoints'] = str([(999, 999), (1999, 1999)])
        log_size = 2999

    evaluation_report = pd.DataFrame()
    for index, row in df.iterrows():
        log_name = row.iloc[0]
        print(f"WIP: {index}: {log_name}")

        # get actual and detected change point infos
        actual_cp = ast.literal_eval(row["Actual Changepoints"])
        detected_cp = [(int(x), int(y)) for x, y in ast.literal_eval(row["Detected Changepoints"])]

        actual_info_rel = [item[0] for item in actual_cp]
        detected_info_rel = [item[0] for item in detected_cp if item[0] == item[1]]

        lag_acc = int(log_size * LAG)

        TP, FP = getTP_FP(detected_info_rel, actual_info_rel, lag_acc)
        FN_TP = len(actual_info_rel)
        evaluation_row = {'log_name': log_name,
                          'complexity': "na",
                          'actual_cp': actual_info_rel,
                          'detected_cp': detected_info_rel,
                          'log_size': log_size,
                          'lag_indices': lag_acc,
                          'TP': TP,
                          'FP': FP,
                          'FN_TP': FN_TP,
                          'lag': LAG,
                           }
        evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                      ignore_index=True)

    return evaluation_report



def reorganize_drift_info(actual_drifts: List[str], actual_cp: List[Tuple[int, int]], detected_drifts: List[str], detected_cp: List[Tuple[int, int]]) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    Reorganizes drift information into a structured format for further processing.

    Parameters:
    - actual_drifts (List[str]): A list of actual drift types.
    - actual_cp (List[Tuple[int, int]]): A list of actual changepoints associated with the drift types.
    - detected_drifts (List[str]): A list of detected drift types.
    - detected_cp (List[Tuple[int, int]]): A list of detected changepoints associated with the drift types.

    Returns:
    - List[Tuple[str, str, Tuple[int, int]]]: A list of tuples containing the group type ('actual' or 'detected'),
      drift type, and associated changepoint.
    """

    reorganized_drift_info = []

    # Create tuples for actual changepoints
    for drift_type, changepoint in zip(actual_drifts, actual_cp):
        reorganized_drift_info.append(("actual", drift_type, changepoint))

    # Create tuples for detected changepoints
    for drift_type, changepoint in zip(detected_drifts, detected_cp):
        reorganized_drift_info.append(("detected", drift_type, changepoint))

    return reorganized_drift_info


def add_simple_drifts_to_complex(drift_info: List[Tuple[str, str, Tuple[int, int]]]) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    Merges simple drifts with complex drifts based on their changepoints.

    Parameters:
    - drift_info (List[Tuple[str, str, Tuple[int, int]]]): A list of tuples where each tuple contains
      the group type ('detected' or 'actual'), drift type, and changepoint.

    Returns:
    - List[Tuple[str, str, Tuple[int, int]]]: A combined list of merged detected and actual drifts.
    """

    # Initialize dictionaries to group by "detected" and "actual"
    grouped_changepoints = {"detected": [], "actual": []}

    # Group by "detected" and "actual"
    for entry in drift_info:
        group_type, drift_type, changepoint = entry
        grouped_changepoints[group_type].append((drift_type, changepoint))

    # Process both "detected" and "actual" groups
    detected_merged = process_group(grouped_changepoints["detected"])
    actual_merged = process_group(grouped_changepoints["actual"])

    # Combine the results
    final_result = [("detected", drift_type, changepoint) for drift_type, changepoint in detected_merged]
    final_result += [("actual", drift_type, changepoint) for drift_type, changepoint in actual_merged]

    return final_result


def process_group(group: List[Tuple[str, Tuple[int, int]]]) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Processes a group of drift instances, separating them into incremental/recurring
    and sudden/gradual types, and merges overlapping changepoints.

    Parameters:
    - group (List[Tuple[str, Tuple[int, int]]]): A list of tuples where each tuple contains
      a drift type ('incremental', 'recurring', 'sudden', or 'gradual') and a changepoint range.

    Returns:
    - List[Tuple[str, Tuple[int, ...]]]: A list of tuples containing drift types and their merged
      changepoint ranges, ensuring unique and sorted values.
    """

    incremental_or_recurring = []
    sudden_or_gradual = []
    result = []

    # Separate the entries into incremental/recurring and sudden/gradual
    for drift_type, changepoint in group:
        if drift_type in ['incremental', 'recurring']:
            incremental_or_recurring.append((drift_type, changepoint))
        else:
            sudden_or_gradual.append((drift_type, changepoint))

    # Merge changepoints
    for c_drift, c_range in incremental_or_recurring:
        merged_range = list(c_range)  # Start with the incremental/recurring range
        found_overlap = False  # Flag to check if we found any overlapping changepoint

        for drift, range in sudden_or_gradual:
            # Check if the sudden/gradual falls within the incremental/recurring range
            if c_range[0] <= range[0] and range[1] <= c_range[1]:
                found_overlap = True  # We found an overlap
                # Merge the changepoint values
                merged_range.extend(range)

        # If we found an overlap, add the merged range
        if found_overlap:
            result.append((c_drift, tuple(sorted(set(merged_range)))))  # Ensure unique and sorted values
        else:
            # If no overlap, keep the original range
            result.append((c_drift, tuple(c_range)))

    # Add standalone sudden/gradual changepoints if they do not overlap with any incremental/recurring
    for drift, range in sudden_or_gradual:
        is_standalone = True  # Flag to check if it is standalone

        for c_drift, c_range in incremental_or_recurring:
            if c_range[0] <= range[0] and range[1] <= c_range[1]:
                is_standalone = False  # It overlaps with an incremental/recurring
                break

        # If it is standalone, add to the result
        if is_standalone:
            result.append((drift, range))

    return result


def match_drift_instances(drift_info: List[Tuple[str, str, Tuple[int, ...]]], lag_accepted: int) -> List[
    Tuple[Tuple[str, str, Tuple[int, ...]], Tuple[str, str, Tuple[int, ...]]]]:
    """
    Matches detected drift instances with actual drift instances based on changepoints.

    Parameters:
    - drift_info (List[Tuple[str, str, Tuple[int, ...]]]): A list of tuples containing
      detected and actual drift instances.
    - lag_accepted (int): The acceptable lag for matching changepoints.

    Returns:
    - List[Tuple[Tuple[str, str, Tuple[int, ...]], Tuple[str, str, Tuple[int, ...]]]]: A list of matched
      tuples, each containing detected and actual drift instances.
    """

    # Separate detected and actual tuples
    detected_dict: Dict[str, List[Tuple[str, str, Tuple[int, ...]]]] = {}
    actual_dict: Dict[str, List[Tuple[str, str, Tuple[int, ...]]]] = {}

    for entry in drift_info:
        drift_type, changepoints = entry[1], entry[2]
        if entry[0] == 'detected':
            if drift_type not in detected_dict:
                detected_dict[drift_type] = []
            detected_dict[drift_type].append((entry[0], drift_type, changepoints))  # Preserve original structure
        else:
            if drift_type not in actual_dict:
                actual_dict[drift_type] = []
            actual_dict[drift_type].append((entry[0], drift_type, changepoints))  # Preserve original structure

    # Prepare to collect matches
    matched_results: List[Tuple[Tuple[str, str, Tuple[int, ...]], Tuple[str, str, Tuple[int, ...]]]] = []

    # Process each drift type
    for drift_type in detected_dict.keys():
        detected_drifts = detected_dict[drift_type]
        actual_drifts = actual_dict.get(drift_type, [])

        matched = []
        matched_detected = []
        matched_actual = []

        # Try to match detected with actual
        for detected in detected_drifts:
            best_match = None
            best_match_distance = 10**10

            for actual in actual_drifts:
                detected_cp = sorted(set(detected[2]))
                actual_cp = sorted(set(actual[2]))
                matched_indices = assign_changepoints(detected_cp, actual_cp, lag_accepted)
                if len(matched_indices) > 0 and abs(matched_indices[0][0] - matched_indices[0][1]) < best_match_distance:
                    best_match_distance = abs(matched_indices[0][0] - matched_indices[0][1])
                    best_match = (detected, actual)

            if best_match:
                matched.append(best_match)
                matched_detected.append(best_match[0])
                matched_actual.append(best_match[1])

        # Update unmatched detected and actual lists
        unmatched_detected = [d for d in detected_drifts if d not in matched_detected]
        unmatched_actual = [a for a in actual_drifts if a not in matched_actual]

        # Add matched tuples to the results
        for detected, actual in matched:
            matched_results.append((detected, actual))

        # Add unmatched detected tuples
        for d in unmatched_detected:
            matched_results.append(d)  # Denote unmatched detected

        # Add unmatched actual tuples
        for a in unmatched_actual:
            matched_results.append(a)  # Denote unmatched actual

    return matched_results


def calculate_tp_fp(matched_results: List[Tuple[Tuple[str, str, Tuple[int, ...]], Tuple[str, str, Tuple[int, ...]]]],
                    lag_accepted: int) -> Tuple[float, int]:
    """
    Calculates true positives and false positives from matched drift instances.

    Parameters:
    - matched_results (List[Tuple[Tuple[str, str, Tuple[int, ...]], Tuple[str, str, Tuple[int, ...]]]]): A list of matched
      tuples containing detected and actual drift instances.
    - lag_accepted (int): The acceptable lag for counting true positives.

    Returns:
    - Tuple[float, int]: A tuple containing the number of true positives and false positives.
    """

    true_positives, false_positives = 0, 0

    # Keep track of unmatched detected indices
    unmatched_detected = set()

    # Process matched results
    for matched in matched_results:
        if len(matched) == 2:  # There is a match
            detected, actual = matched
            detected_cp = sorted(set(detected[2]))
            actual_cp = sorted(set(actual[2]))
            matched_indices = assign_changepoints(detected_cp, actual_cp, lag_accepted)
            num_matched_indices = len(matched_indices)
            num_actual_indices = len(actual_cp)
            true_positives += num_matched_indices / num_actual_indices  # Calculate TP as a fraction
        else:
            if matched[0] == "detected":
                unmatched_detected.add(matched[0])

    # Count false positives
    false_positives = len(unmatched_detected)
    return true_positives, false_positives


def process_drift_level(row: pd.Series, drift_info_df_noise: pd.DataFrame, LAG: int) -> List[Dict[str, Any]]:
    """
    Processes the drift level for a given log entry and evaluates the drift types.

    Parameters:
    - row (pd.Series): A row from the DataFrame containing log information.
    - drift_info_df_noise (pd.DataFrame): DataFrame containing noise information for drift levels.
    - LAG (int): The lag value used in processing drift levels.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each drift type.
    """

    log_name = row.iloc[0]  # Get log name from the first column
    log_size = row['n_traces']  # Get log size
    lag_acc = int(log_size * LAG)  # Calculate lag accumulator

    # Select noise level used for the given log
    noise_level = get_noise_info(log_name, drift_info_df_noise)

    # Create evaluation dictionary for drift types
    drift_types = ['no_drift', "incremental", "recurring", "sudden", "gradual"]

    # Get actual and detected change point info
    actual_cp = ast.literal_eval(row["Actual Changepoints"])
    actual_drift = ast.literal_eval(row["Actual Drift Types"])
    detected_cp = ast.literal_eval(row["Detected Changepoints"])
    detected_drift = ast.literal_eval(row["Predicted Drift Types"])

    # Reorganize drift information
    drift_info_reorganized = reorganize_drift_info(actual_drift, actual_cp, detected_drift, detected_cp)

    # add no_drift items if no detected/actual drifts are detected
    if all(drift[0] == 'actual' for drift in drift_info_reorganized):
        drift_info_reorganized.append(("detected", "no_drift", (0,0)))
    if all(drift[0] == 'detected' for drift in drift_info_reorganized):
        drift_info_reorganized.append(("actual", "no_drift", (0, 0)))
    #rel_drifts_agg = add_simple_drifts_to_complex(drift_info_reorganized)

    evaluation_rows = [] # Initialize a list to hold evaluation results
    for drift_type in drift_types:
        rel_drifts = [drift for drift in drift_info_reorganized if drift[1] == drift_type]
        # Initialize True Positive, False Positive, and FN_TP
        rel_drifts_agg_matched = match_drift_instances(rel_drifts, lag_acc)
        TP, FP = calculate_tp_fp(rel_drifts_agg_matched, lag_acc)
        FN_TP = len([drift for drift in rel_drifts if drift[0] == 'actual'])

        # Extract and sort actual and detected change points
        actual_cp_sorted = [drift[-1] for drift in rel_drifts if (drift[1] == drift_type and drift[0] == 'actual')]
        actual_cp_sorted = sorted(actual_cp_sorted, key=lambda x: x[0])

        detected_cp_sorted = [drift[-1] for drift in rel_drifts if (drift[1] == drift_type and drift[0] == 'detected')]
        detected_cp_sorted = sorted(detected_cp_sorted, key=lambda x: x[0])

        # Create the evaluation row dictionary
        evaluation_row = {
            'log_name': log_name,
            'noise_level': noise_level,
            'complexity': "na",  # Complexity is not calculated in this function
            'drift_type': drift_type,
            'actual_cp': actual_cp_sorted,
            'detected_cp': detected_cp_sorted,
            'log_size': log_size,
            'lag': LAG,
            'lag_indices': lag_acc,
            'TP': TP,
            'FP': FP,
            'FN_TP': FN_TP
        }
        evaluation_rows.append(evaluation_row)  # Append the evaluation row to the list

    return evaluation_rows


def evaluation_drift_level_in_parallel(full_path: str, drift_info_path:str, LAG: int, num_cores: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates drift levels in parallel processing using multiple CPU cores.

    Parameters:
    - full_path (str): The file path to the CSV file containing evaluation information.
    - LAG (int): The lag value used in processing drift levels.
    - num_cores (Optional[int]): The number of CPU cores to use for parallel processing. Defaults to None, which uses all available cores.

    Returns:
    - pd.DataFrame: A DataFrame containing the evaluation report of drift levels.
    """

    # Load CSV file with evaluation info
    df = pd.read_csv(full_path)

    # Load initial drift info file
    drift_info_df = load_initial_drift_info_file(drift_info_path)
    drift_info_df_noise = drift_info_df.loc[drift_info_df["drift_sub_attribute"] == "noisy_trace_prob"]

    # Prepare the evaluation report DataFrame
    evaluation_report = pd.DataFrame()

    # Total number of tasks to process
    total_tasks = len(df)

    # Use ProcessPoolExecutor to parallelize the processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks for parallel execution and store results as futures
        futures = [executor.submit(process_drift_level, row, drift_info_df_noise, LAG) for _, row in df.iterrows()]

        # Initialize a counter for completed tasks
        completed_tasks = 0

        # Collect results from futures as they complete and concatenate into the DataFrame
        for future in futures:
            evaluation_rows = future.result()  # Expecting a list of evaluation rows
            evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records(evaluation_rows)],
                                          ignore_index=True)

            # Increment the completed tasks counter
            completed_tasks += 1

            # Calculate and print progress, overwriting the same line
            progress_percentage = (completed_tasks / total_tasks) * 100
            sys.stdout.write(f"\rProgress drift-level calculations: {progress_percentage:.2f}% completed.")
            sys.stdout.flush()  # Ensure the output is written immediately

    print()  # Print a new line after completion
    return evaluation_report


def process_change_points_level(row: pd.Series,
                                drift_info_df_noise: pd.DataFrame,
                                LAG: int) -> Dict[str, Any]:
    """
    Processes a row of change point data and evaluates detected change points against actual change points.

    Parameters:
    - row (pd.Series): A row from the DataFrame containing log evaluation data.
    - drift_info_df_noise (pd.DataFrame): DataFrame containing noise information for drift analysis.
    - LAG (int): The lag value used for evaluating change points.

    Returns:
    - Dict[str, Any]: A dictionary containing the evaluation results, including log name, noise level,
                      actual and detected change points, and true/false positives.
    """

    log_name = row.iloc[0]  # Assuming the first element is the log name
    log_size = row['n_traces']  # Total number of traces in the log
    lag_acc = int(log_size * LAG)  # Calculate lag accuracy

    # Select noise level used for the given log
    noise_level = get_noise_info(log_name, drift_info_df_noise)

    # Get actual and detected change point info
    actual_cp: List[int] = ast.literal_eval(row["Actual Changepoints"])
    actual_drift: List[str] = ast.literal_eval(row["Actual Drift Types"])

    detected_cp: List[int] = ast.literal_eval(row["Detected Changepoints"])
    detected_drift: List[str] = ast.literal_eval(row["Predicted Drift Types"])

    drift_info_reorganized: List[Tuple[str, str, Tuple[int, int]]] = reorganize_drift_info(actual_drift, actual_cp,
                                                                                           detected_drift, detected_cp)

    # Initialize empty sets for actual and detected change points
    actual_change_points_set = set()
    detected_change_points_set = set()

    # Iterate through the data to extract change points
    for category, _, change_point in drift_info_reorganized:
        if category == 'actual':
            actual_change_points_set.update(change_point)  # Use update for sets
        elif category == 'detected':
            detected_change_points_set.update(change_point)  # Use update for sets

    # Convert sets to sorted lists
    actual_change_points = sorted(actual_change_points_set)
    detected_change_points = sorted(detected_change_points_set)

    # Calculate true positives and false positives
    TP, FP = getTP_FP(detected_change_points, actual_change_points, lag_acc)
    FN_TP = len(actual_change_points)  # Number of actual change points

    # Prepare the evaluation row as a dictionary
    evaluation_row: Dict[str, Any] = {
        'log_name': log_name,
        'noise_level': noise_level,
        'drift_type': 'NA',
        'actual_cp': actual_change_points,
        'detected_cp': detected_change_points,
        'log_size': log_size,
        'lag': LAG,
        'lag_indices': lag_acc,
        'TP': TP,
        'FP': FP,
        'FN_TP': FN_TP
    }

    return evaluation_row



def evaluation_change_point_level_in_parallel(full_path: str, drift_info_path:str, LAG: int,
                                              num_cores: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates change point levels in parallel processing using multiple CPU cores.

    Parameters:
    - full_path (str): The file path to the CSV file containing evaluation information.
    - LAG (int): The lag value used in processing change points.
    - num_cores (Optional[int]): The number of CPU cores to use for parallel processing. Defaults to None, which uses all available cores.

    Returns:
    - pd.DataFrame: A DataFrame containing the evaluation report of change points.
    """

    # Load CSV file with evaluation info
    df = pd.read_csv(full_path)

    # Load initial drift info file
    drift_info_df = load_initial_drift_info_file(drift_info_path)
    drift_info_df_noise = drift_info_df.loc[drift_info_df["drift_sub_attribute"] == "noisy_trace_prob"]

    # Prepare the evaluation report DataFrame
    evaluation_report = pd.DataFrame()

    # Total number of tasks to process
    total_tasks = len(df)

    # Use ProcessPoolExecutor to parallelize the processing of logs using multiple CPU cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks for parallel execution and store results as futures
        futures = [executor.submit(process_change_points_level, row, drift_info_df_noise, LAG) for _, row in
                   df.iterrows()]

        # Initialize a counter for completed tasks
        completed_tasks = 0

        # As futures are completed, collect results and concatenate into the DataFrame
        for future in futures:
            evaluation_row = future.result()
            evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                          ignore_index=True)

            # Increment the completed tasks counter
            completed_tasks += 1

            # Calculate and print progress, overwriting the same line
            progress_percentage = (completed_tasks / total_tasks) * 100
            sys.stdout.write(f"\rProgress change point level: {progress_percentage:.2f}% completed.")
            sys.stdout.flush()  # Ensure the output is written immediately

    print()  # Print a new line after completion
    return evaluation_report



def evaluate_results_per_lag_cp(lag, base_path_main, num_cores):
    """
    Process a specific LAG value by generating evaluation reports for drift and change points.

    Parameters:
    - lag: The relative lag value to process.
    - base_path_main: The base directory where the evaluation files are stored.
    - num_cores: Number of cores to use for parallel processing.

    Returns:
    - evaluation_report_drift: DataFrame containing the drift evaluation results.
    - evaluation_report_cp: DataFrame containing the change point evaluation results.
    - elapsed_time: Time taken to process the current lag.
    """
    lag_start_time = time.time()

    file_name = f"evaluation_results_general_{lag}_lag.csv"
    full_path = os.path.join(base_path_main, file_name)

    # Generate evaluation reports using parallel processing
    evaluation_report_cp = evaluation_change_point_level_in_parallel(full_path, LAG=lag, num_cores=num_cores)

    # Log time taken for this lag
    elapsed_time = time.time() - lag_start_time
    print(f"Time taken for LAG={lag}: {elapsed_time:.2f} seconds")

    return evaluation_report_cp, elapsed_time


def evaluate_results_per_lag(lag, drift_info_path, base_path_main, num_cores):
    """
    Process a specific LAG value by generating evaluation reports for drift and change points.

    Parameters:
    - lag: The relative lag value to process.
    - base_path_main: The base directory where the evaluation files are stored.
    - num_cores: Number of cores to use for parallel processing.

    Returns:
    - evaluation_report_drift: DataFrame containing the drift evaluation results.
    - evaluation_report_cp: DataFrame containing the change point evaluation results.
    - elapsed_time: Time taken to process the current lag.
    """
    lag_start_time = time.time()

    file_name = f"evaluation_results_general_{lag}_lag.csv"
    full_path = os.path.join(base_path_main, file_name)

    # Generate evaluation reports using parallel processing
    evaluation_report_cp = evaluation_change_point_level_in_parallel(full_path, drift_info_path, LAG=lag, num_cores=num_cores)
    evaluation_report_drift = evaluation_drift_level_in_parallel(full_path, drift_info_path, LAG=lag, num_cores=num_cores)

    # Log time taken for this lag
    elapsed_time = time.time() - lag_start_time
    print(f"Time taken for LAG={lag}: {elapsed_time:.2f} seconds")

    return evaluation_report_drift, evaluation_report_cp, elapsed_time


def save_reports(evaluation_report, base_path_main, file_name, ):
    """
    Save the drift and change point evaluation reports to CSV files.

    Parameters:
    - evaluation_report_drift: DataFrame containing the drift evaluation results.
    - evaluation_report_cp: DataFrame containing the change point evaluation results.
    - base_path_main: The base directory where the reports will be saved.
    """
    report_path = os.path.join(base_path_main, file_name)
    evaluation_report.to_csv(report_path, index=True)
    print(f"Reports saved: {report_path}")


def main(drift_info_path:str, base_path_main: str, num_cores: int):
    """
    Main function to process all relative lags and generate the evaluation reports.

    Parameters:
    - base_path_main (str): The base directory where the evaluation files are stored.
    - num_cores (int): Number of cores to use for parallel processing.
    """
    # Start tracking total time
    start_time = time.time()

    # Initialize lists to store reports for each lag
    combined_evaluation_report_drift = []
    combined_evaluation_report_cp = []

    for lag in RELATIVE_LAG:
        print(f"Lag in progress: {lag}")

        # Get the evaluation reports for the current lag
        evaluation_report_drift, evaluation_report_cp, elapsed_time = evaluate_results_per_lag(lag, drift_info_path, base_path_main,
                                                                                               num_cores)

        # Add lag as a new column to the reports
        evaluation_report_drift['lag'] = lag
        evaluation_report_cp['lag'] = lag

        # Append current lag's results to the combined lists
        combined_evaluation_report_drift.append(evaluation_report_drift)
        combined_evaluation_report_cp.append(evaluation_report_cp)

        print(f"Elapsed time for lag {lag}: {elapsed_time:.2f} seconds\n")

    # Combine the reports into single DataFrames
    final_evaluation_report_drift = pd.concat(combined_evaluation_report_drift, ignore_index=True)
    final_evaluation_report_cp = pd.concat(combined_evaluation_report_cp, ignore_index=True)

    # Save the reports
    save_reports(final_evaluation_report_drift, base_path_main, 'evaluation_results_level_drift.csv')
    save_reports(final_evaluation_report_cp, base_path_main, 'evaluation_results_level_change_points.csv')

    # Aggregate reports
    grouping = [["lag"], ["noise_level"]]
    summarize_results(final_evaluation_report_drift, base_path_main, grouping, 'evaluation_results_aggregated_level_drift.csv')
    grouping = [["lag"], ["lag", "drift_type"], ["lag", "noise_level"]]
    summarize_results(final_evaluation_report_cp, base_path_main, grouping,'evaluation_results_aggregated_level_cp.csv')


    # Log total time taken
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"Total time taken: {hours} hours, {minutes} minutes, {seconds} seconds")




def summarize_results(results_df, base_path_main, grouping, file_name):

    for group in grouping:
        result = get_total_evaluation_results(results_df, group)

        # Pivot the DataFrame
        df_pivoted = result[['Precision', 'Recall', 'F1_score']].unstack(level=group[0])

        if len(group) == 2:
            # Sort the columns
            df_pivoted.columns = df_pivoted.columns.swaplevel(0, 1)
            df_pivoted = df_pivoted.sort_index(level=0, axis=1)

            new_order = []
            for lag in sorted(set(df_pivoted.columns.get_level_values(0))):  # Get unique lags and sort them
                new_order.append((lag, 'Precision'))
                new_order.append((lag, 'Recall'))
                new_order.append((lag, 'F1_score'))

            # Create the new MultiIndex
            new_index = pd.MultiIndex.from_tuples(new_order, names=[group[0], None])
            df_pivoted = df_pivoted[new_index]


            if group[1] == "drift_type":
                index = ['no_drift', 'sudden', 'gradual', 'incremental', 'recurring']
                df_pivoted = pd.DataFrame(df_pivoted, index=index)

            elif group[1] == "noise_level":
                index = ['0.0', '0.3', '0.6']
                df_pivoted = pd.DataFrame(df_pivoted, index=index)
            else:
                pass
        else:
            # Reset the index to convert the Series to DataFrame
            df_reset = df_pivoted.reset_index()

            # Rename columns for clarity
            df_reset.columns = ['Metric', 'lag', 'Value']

            # Set the new index with lag and Metric
            df_swapped = df_reset.set_index(['lag', 'Metric'])

            # Sort the index levels
            df_swapped = df_swapped.sort_index(level=['lag', 'Metric'])

            new_order = []
            for lag in set(df_pivoted.index.get_level_values(1)):  # Get unique lags and sort them
                new_order.append((lag, 'Precision'))
                new_order.append((lag, 'Recall'))
                new_order.append((lag, 'F1_score'))

            sorted_tuples = sorted(new_order, key=lambda x: (x[0]))

            # Create a MultiIndex from the new_index list
            new_multi_index = pd.MultiIndex.from_tuples(sorted_tuples)

            df_pivoted = df_swapped.reindex(new_multi_index).T


        focus = "_".join(group)
        file_name_new = file_name.split('.')[0] + '_' + focus+ ".csv"
        report_path = os.path.join(base_path_main, file_name_new)
        df_pivoted.to_csv(report_path, index=True)
        print(f"Aggregated reports saved {focus}: {report_path}")
        print(df_pivoted)
        # Round values to 2 decimal places
        df_rounded = df_pivoted.round(2)

        # Convert to LaTeX format
        latex_output = df_rounded.to_latex(index=True, header=True, float_format="%.2f")

        print(latex_output)
        print()




def log_total_time(start_time: float):
    """
    Logs the total time taken for processing.

    Parameters:
    - start_time (float): The starting time of the process.
    """
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"Total time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


