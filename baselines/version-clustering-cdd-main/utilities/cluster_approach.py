"""Main pipeline for the clustering-based concept drift detection approach: for
each event log, builds windowed DFG similarity vectors, clusters windows with
HDBSCAN into process versions, detects change points via ruptures on the cluster
probabilities, and writes prediction/annotation/visualization output files."""

# Standard library
import os
import json
import multiprocessing
from functools import partial
import ast
import time
from itertools import product

# Data manipulation
import pandas as pd
import numpy as np

# Scientific / ML libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import hdbscan
import ruptures as rpt

# Progress bars
from tqdm import tqdm

# PM4Py

# Custom utilities
import utilities.utilities as utils
import utilities.preprocessing_methods as ppm

#############################################################################################################
############################################### PIPELINE ####################################################
#############################################################################################################


def _run_processing_pipeline(log_dir, drift_info, n_windows, min_cluster_size, threshold, pen,
                             track_algorithm_results=False, dataset_name=None,
                             process_fn=None, algorithm_name="Bernold"):
    """Shared body of processing_pipeline_cdlg and processing_pipeline_general.

    Args:
        log_dir: Path of the directory where all the data is stored.
        drift_info: Pre-loaded drift info DataFrame, or None if no ground truth is available.
        n_windows (int): Number of windows.
        track_algorithm_results (bool): Whether to also build and save the algorithm_results.csv
            comparison file (only used by processing_pipeline_general).
        dataset_name: Name used for the "Log Source" column when track_algorithm_results is True.
        process_fn: Function that processes one log (defaults to process_log).
        algorithm_name: Name used for the "Algorithm" column when track_algorithm_results is True.
    """
    if process_fn is None:
        process_fn = process_log

    ground_truth_available = drift_info is not None

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(log_dir)

    # Create X_set and y_set directory to store all the inputs for the ML model.
    utils.make_dir(f"{log_dir}/outputs")

    # Creates annotations directory for after preprocessing info.
    dir_annotations = f"{log_dir}/Annotations"
    utils.make_dir(dir_annotations)

    # Dictionaries to be saved as annotations for after preprocessing info.
    window_info = {}
    date_info = {}
    number_of_traces = {}
    algorithm_results = None
    if track_algorithm_results:
        algorithm_results = {
            "Algorithm": [],
            "Log Source": [],
            "Log": [],
            "Detected Changepoints": [],
            "Actual Changepoints for Log": [],
            "Duration (Seconds)": [],
            "Duration": [],
            "Window Size": [],
        }

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    # Process logs in parallel using the n-1 cores of the cpu. Creates all the X_set files
    with multiprocessing.Pool(num_cores) as pool:
        partial_process_log = partial(process_fn,
                                      drift_info=drift_info,
                                      n_windows=n_windows,
                                      min_cluster_size=min_cluster_size,
                                      threshold=threshold,
                                      pen=pen)
        results = list(tqdm(pool.imap(partial_process_log, log_files.items()),
                            desc="Preprocessing Event Logs", unit=" Event Log"))

    # Process the drift info file. Creates all the y_set files.
    for result in tqdm(results, desc="Finishing preprocessing of Event Logs"):
        name, pred_drift, real_drift, execution_time, borders, window_information, log_date_info, n_of_traces = result
        # In case there was an error processing the log. Everything will be None.
        if name != None:

            if ground_truth_available:
                utils.update_trace_indices(drift_info, name, borders)
                drift_info.loc[drift_info['log_name'] == name, 'time_seconds'] = execution_time

            # Updates annotation info.
            window_info[name] = window_information
            date_info[name] = log_date_info
            number_of_traces[name] = n_of_traces

            if track_algorithm_results:
                # Update algorithm_results
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                if ground_truth_available:
                    pred_drift = get_changepoints_trace_idx(pred_drift, window_information, n_windows)
                    real_drift = get_changepoints_trace_idx(real_drift, window_information, n_windows)
                name = name.split(".")[0]

                algorithm_results["Algorithm"].append( algorithm_name )
                algorithm_results["Log Source"].append( dataset_name)
                algorithm_results["Log"].append(name)
                algorithm_results["Detected Changepoints"].append(pred_drift)
                algorithm_results["Actual Changepoints for Log"].append( real_drift)
                algorithm_results["Duration (Seconds)"].append( execution_time)
                algorithm_results["Duration"].append( formatted_time)
                algorithm_results["Window Size"].append( n_windows )

    # Print total of event log were actually processed
    print("Event logs actually processed: ", len(number_of_traces.keys()) )
    print("Event logs with errors, no processed: ", len(log_files.keys()) - len(number_of_traces.keys()) )

    # After preprocessing creating annotation files.
    if ground_truth_available:
        drift_info = drift_info[drift_info["log_name"].isin(number_of_traces.keys())] # Makes a of only the preprocessed logs.
        drift_info.to_csv(os.path.join(dir_annotations, "drift_info_updated_windows.csv"))

    window_info_path = os.path.join(dir_annotations, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)

    date_info_path = os.path.join(dir_annotations, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    number_of_traces_path = os.path.join(dir_annotations, "number_of_traces.json")
    with open(number_of_traces_path, "w", encoding='utf-8') as file:
        json.dump(number_of_traces, file)

    if track_algorithm_results:
        algorithm_results_df = pd.DataFrame(algorithm_results)
        algorithm_results_df.to_csv(f"{dir_annotations}/algorithm_results.csv", index=False)

    return log_files, drift_info, window_info


def processing_pipeline_cdlg(log_dir,n_windows=100,min_cluster_size=10,threshold=0.5,pen=1):
    """Main function for preprocessing the event logs.

    Args:
        log_dir: Path of the directory where all the data is stored.
        n_windows (int, optional): Number of windows. Defaults to 100.
    """

    drift_info = utils.extract_drift_information(log_dir)
    drift_info["time_seconds"] = 0 # Adds an execution time column

    return _run_processing_pipeline(log_dir, drift_info, n_windows, min_cluster_size, threshold, pen)


def processing_pipeline_general(log_dir,n_windows=100,min_cluster_size=10,threshold=0.5,pen=1):
    """Main function for processing the event logs

    Args:
        log_dir: Path of the directory where all the data is stored.
        n_windows (int, optional): Number of windows. Defaults to 100.
    """

    dataset_name = log_dir.split("/")[-1]

    drift_info = None
    if os.path.exists(f"{log_dir}/drift_info.csv"):
        drift_info = load_drift_info(f"{log_dir}/drift_info.csv")
        drift_info["time_seconds"] = 0 # Adds an execution time column

    return _run_processing_pipeline(log_dir, drift_info, n_windows, min_cluster_size, threshold, pen,
                                    track_algorithm_results=True, dataset_name=dataset_name)


def load_drift_info(log_dir):
    drift_info = pd.read_csv(log_dir)
    drift_info["log_name"] = drift_info["log_name"].astype(str)
    drift_info["drift_traces_index"] = drift_info["drift_traces_index"].apply(lambda x: list(map(int, ast.literal_eval(x))))
    drift_info["change_trace_index"] = drift_info["change_trace_index"].apply(lambda x: list(map(int, ast.literal_eval(x))))
    return drift_info
    

#############################################################################################################
############################################# PROCESS LOG ###################################################
#############################################################################################################

def process_log(name_path,drift_info, n_windows,min_cluster_size,threshold,pen):
    """Function for processing the raw event logs.
    It creates a numpy file with the extracted information for a single event log. 
    This file will be the input for the ML model.

    Args:
        log_dir: Path of the directory where all the data is stored. 
        n_windows (int, optional): Number of windows. Defaults to 100.
    """
    try:
        start_time = time.time()       # Record start

        # Step 1: Measuring Similarity Across Windows
        
        name, path = name_path 
        #drift_number = int(name.split("_")[1])
        
        # load event log
        event_log = utils.import_event_log(path=path, name=name)
        
        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)
        
        number_of_traces = utils.get_number_of_traces(filtered_log)
        
        # Create similarity vectors
        dfg, borders, window_information, log_date_info = ppm.log_to_windowed_dfg_count(filtered_log, n_windows)
        similarity = ppm.similarity_calculation(np.array(dfg))
        
        if similarity.shape[0] == 0:
            raise ValueError("Empty dataset passed to cluster_hdbscan")


        # Step 2: Clustering Windows into Process Versions
        
        # Create clusters and find change points
        probabilities, clusters_labels = cluster_hdbscan(similarity,min_cluster_size=min_cluster_size,threshold=threshold)


        # Step 3: Change Point Detection
        pred_drift = find_transition_steps(probabilities,pen,n_windows)

        # Step 4: create output files
        
        # Get real drift points
        real_drift = []
        if drift_info is not None:
            real_drift = process_drift(drift_info, name, borders)
            
        # Save results
        
        name_no_extention = name.split(".")[0] # removes the .xes
        np.savetxt(f"{path}/outputs/{name_no_extention}.txt", pred_drift, fmt='%d')
        filename_hdbscan = f"{path}/outputs/{name_no_extention}_output.svg"
        save_plot_drift(clusters_labels,real_drift,pred_drift,filename_hdbscan)
        
        # Create output for visualization framework
        output_one, output_two = to_vis_system(dfg,n_windows,clusters_labels,pred_drift,window_information)
        output_one.to_csv(f"{path}/outputs/{name_no_extention}_general.csv",index=False)
        output_two.to_csv(f"{path}/outputs/{name_no_extention}_df_info.csv",index=False)
        
        end_time = time.time()         # Record end
        execution_time = end_time - start_time
        
        return name, pred_drift, real_drift, execution_time, borders, window_information, log_date_info, number_of_traces
    except:
        # In case of an errror happens when reading the event log. 
        #print(name)
        return None, None, None, None, None, None, None, None



def to_vis_system(dfg_list,n_windows,clusters_labels,pred_drift,window_information):

    #################################################
    # 1: Create output data for the first file needed.
    #################################################

    pred_drift = set(pred_drift)
    output_one = []
    for i in range(n_windows):
        timestamp = window_information[i+1][1] # i+1 because it starts in 1 
                                               # and then 1 because format is ('trace ID', ' Timestamp format: %Y-%m-%d %H:%M:%S')
        version = clusters_labels[i]
        drift = 1 if i+1 in pred_drift else 0
        output_one.append([i,timestamp,version,drift])
    output_one = pd.DataFrame(output_one, columns=["window_id", "timestamp", "version", "drift"])

    #################################################
    # 2: Create output data for the second file needed.
    #################################################
    
    activities = list(dfg_list[0].columns)

    # Create dictionary with each 2-combination as a key and empty value (or 0)
    pair_dict = {(a, b): [0] * n_windows for a, b in product(activities, repeat=2)}
    
    i = 0 # Window index
    for dfg in dfg_list:
        # For the dfg of each window, populate the pair_dict
        for row_label, row in dfg.iterrows():
            for col_label, value in row.items():
                pair_dict[(row_label, col_label)][i] += value

        i+=1 # update window index

    output_two = pd.DataFrame([
        {'act1': a, 'act2': b, **{i: val for i, val in enumerate(values)}}
        for (a, b), values in pair_dict.items()
    ])
    return output_one, output_two
    

def save_plot_drift(clusters, real_drift, pred_drift, filename):
    # --- Plot temporal evolution ---
    plt.figure(figsize=(10, 2))  # make plot shorter vertically
    plt.scatter(range(len(clusters)), clusters, color='black', s=45)  # fixed color
    plt.xlabel("Window")
    plt.ylabel("Version ID", labelpad=5)
    plt.title("Temporal evolution of event log", pad=10)
    
    # Compact the y-axis
    plt.ylim(min(clusters) - 0.2, max(clusters) + 0.2)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # show only integer ticks

    # Add vertical lines for real drifts
    for i, step in enumerate(real_drift):
        plt.axvline(
            x=step, color='black', linestyle='--', alpha=0.7,
            label='real drift' if i == 0 else ""
        )
        plt.text(step + 0.2, max(clusters) + 0.2,  # lowered label a bit
                 f"{step}", rotation=90, color='black', fontsize=7,
                 va='bottom', ha='center')
    
    # Add vertical lines for predicted drifts
    for i, step in enumerate(pred_drift):
        plt.axvline(
            x=step, color='red', linestyle='-', alpha=0.7,
            label='pred drift' if i == 0 else ""
        )
        plt.text(step + 0.2, max(clusters) + 0.2,
                 f"{step}", rotation=90, color='red', fontsize=7,
                 va='bottom', ha='center')
    
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', format="svg")
    plt.close()

def process_drift(drift_info, name, borders ):
    """Function for processing the drift info.
    It takes information from the drift_info.csv file and for the individually processed logs
    and creates a numpy array file with the information of the windows where there is a drift.
    This file will be the output for the ML model.

    Args:
        log_dir: Path of the directory where all the data is stored. 
        n_windows (int, optional): Number of windows. Defaults to 100.
    """
    # Get the window information to update the drift_info
    drift_info = utils.update_trace_indices(drift_info, name, borders)
    
    # Creates a view to get inly the drift from 1 specific Event log
    df_filtered = drift_info.loc[drift_info["log_name"] == name]

    # Flatten and get unique values
    unique_values = sorted({x-1 for sublist in df_filtered['change_trace_index'] for x in sublist})
    
    return unique_values

def get_changepoints_trace_idx(window_idx: list, window_info: dict, n_windows : int):
    """Get changepoints as trace indices for WINSIM encoding. 

    Args:
        window_idx (list): List of window indices where a drift happened
        window_info (dict): Window info

    Returns:
        List[tuple]: List of changepoints as trace indices
    """

    change_points = []
    if len(window_idx) == 0:
        return change_points
    else:
        for i in range(len(window_idx)): 
            
            change_point = window_idx[i]
            if change_point > n_windows:
                change_point = n_windows
            elif change_point < 1:
                change_point = 1
            change_point_trace_id = int(window_info[int(change_point)][0])
                
            change_points.append(change_point_trace_id)
        return change_points

#############################################################################################
###################################  CLUSTERING  ############################################
#############################################################################################

def find_transition_steps(probabilities,pen,n_windows):
    # Step 3: Change point detection (on embeddings)
    algo = rpt.KernelCPD(kernel="rbf").fit(probabilities)
    change_points = algo.predict(pen=pen)
    
    # Step 4: Interpret with cluster info
    change_points = [v for v in change_points if v < n_windows]

    return change_points

def order_clusters(clusters):
    # --- Relabel clusters by first appearance ---
    unique_in_order = []
    label_map = {-1:0}
    
    for c in clusters:
        if c not in label_map:
            label_map[c] = len(label_map)  # assign next available label
        unique_in_order.append(label_map[c])
    
    return np.array(unique_in_order)


def cluster_hdbscan(X,min_cluster_size=10,threshold=0.5):
    # X: your dataset as a 2D numpy array
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    clusterer.fit(X)
    
    # Soft cluster membership probabilities
    probabilities = hdbscan.all_points_membership_vectors(clusterer)
    
    # Assign cluster only if probability >= threshold
    labels = []
    for i, probs in enumerate(probabilities):
        max_prob = probs.max()
        if max_prob >= threshold:
            cluster = probs.argmax()
            labels.append(cluster)
        else:
            labels.append(-1)  # -1 indicates noise / unassigned
    
    return probabilities, order_clusters(labels)
        

