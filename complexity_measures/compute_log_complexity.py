# compute a complexity vector per window. Output: .npy/.json vector files + a heatmap per log.
import sys
import os                       # lists every .xes file in a folder automatically

import pm4py                    # the core process mining library, reads and processes event logs

import numpy as np              # turns the list of vectors into a matrix and saves to disk
import json                     # writes window/border info to disk as readable files
import matplotlib.pyplot as plt # draws the heatmap at the end

from log_diversity_measures.diversity_measures import DiversityMeasures
from log_diversity_measures.species_log_representation import SpeciesLogRepresentation

# Imported at module level so that compute_log_complexity_sweep.py can import
# process_single_log from this file.
from windowing import import_event_log, filter_complete_events, window_event_log

# This list documents every measure the library can compute
supported_complexity_measures = [
    "average_distinct_events_per_trace",
    "deviation_from_random",
    "enriched_normalized_sequence_entropy",
    "enriched_normalized_variant_entropy",
    "enriched_sequence_entropy",
    "enriched_variant_entropy",
    "lempel_ziv_complexity",
    "normalized_sequence_entropy",
    "normalized_variant_entropy",
    "num_acyclic_paths",
    "num_event_types",
    "num_events",
    "num_sequences",
    "num_ties_in_transition_matrix",
    "num_unique_sequences",
    "pct_unique_sequences",
    "sequence_entropy",
    "trace_length_average",
    "trace_length_max",
    "trace_length_min",
    "variant_entropy",
    "coverage",
    "coverage_alternative",
    "estimate_asymptotic_exp_shannon_entropy_abundance",
    "estimate_asymptotic_simpson_diversity_abundance",
    "estimate_asymptotic_species_richness_abundance",
    "exp_shannon_entropy_abundance",
    "simpson_diversity_abundance",
    "species_richness_abundance",
]


def compute_complexity_vector(log):
    # computes one complexity vector for a single window
    complexity_profile = DiversityMeasures(log)                   # set up the measurement engine for this window
    complexity_profile.register_n_tupel_species(2)                # treat every 2-activity sequence as a distinct "species"
    complexity_profile.register_n_tupel_species(3)                # treat every 3-activity sequence as a distinct "species"
    complexity_profile.deregister_measure("avg_edit_distance")    # excludes this measure from the computation
    complexity_profile.deregister_measure("num_acyclic_paths")    # excludes this measure from the computation as well

    complexity_vector = []   # collects every number we compute for this window
    measure_names = []       # collects the matching name for each number

    # measures that don't need n-grams, just run each one and store the result
    for k, v in complexity_profile.registered_measures.items():
        measure = v()                        # runs the measure and get its value
        complexity_vector.append(measure)
        measure_names.append(k)

    # measures that DO need n-grams -- run once for 2-gram, once for 3-gram
    for k, v in complexity_profile.species_dependent_measures.items():
        species_representation = complexity_profile.parametrized_representations[SpeciesLogRepresentation.N_GRAM_SPECIES]
        for key in species_representation.keys():
            measure = v(species_representation[key])
            complexity_vector.append(measure)
            measure_names.append(f"{k} (n={key})") # produces names like "measure_name (n=2)" and "measure_name (n=3)"

    # returned so the caller can stack them into a matrix, save them to disk and plot them.
    return complexity_vector, measure_names


def plot_complexity_vectors(complexity_vectors, measure_names, output_path, title="Complexity vector sequence across windows (normalized per measure)"):
    # complexity_vectors has shape (n_windows, 53) -- one row per window, one column per measure.

    # keepdims=True keeps the result as a (1, 53) array instead of flattening it, so the subtraction
    # below still lines up correctly against the full (n_windows, 53) matrix.
    row_min = complexity_vectors.min(axis=0, keepdims=True)
    row_max = complexity_vectors.max(axis=0, keepdims=True)
    # row_min/row_max here are actually COLUMN min/max (axis=0 means "collapse the window axis").

    # Rescale every measure into the 0-1 range, using ONLY this log's own window values
    normalized = (complexity_vectors - row_min) / (row_max - row_min + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 12))  # Creates figure

    im = ax.imshow(normalized.T, aspect="auto", cmap="viridis")
    # imshow draws a matrix as a grid of colored cells. We pass normalized.T (transposed) instead
    # of normalized directly, because imshow puts the matrix's ROWS on the y-axis and COLUMNS on
    # the x-axis -- our original matrix has windows as rows and measures as columns, but we want
    # the opposite on screen (measures on y, windows on x), so we flip it with .T.
    # aspect="auto" lets each cell be rectangular instead of forcing perfect squares.
    # cmap="viridis" is the color scheme: dark purple = low (near 0), yellow = high (near 1).

    ax.set_xlabel("Window")
    ax.set_ylabel("Complexity measure")

    # Force one tick per measure (0 to 52) and label each tick with the actual measure name,
    # instead of matplotlib picking its own tick spacing -- otherwise most of the 53 names
    # would be skipped/hidden. fontsize=6 is small on purpose, so all 53 names fit without
    # overlapping each other.
    ax.set_yticks(range(len(measure_names)))
    ax.set_yticklabels(measure_names, fontsize=6)
    ax.set_title(title) # shows the log's numbered label 
    fig.colorbar(im, ax=ax, label="Normalized value (0-1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def process_single_log(dataset_folder, log_filename, log_number, n_windows=10, output_subfolder=None):
    log_name = log_filename.split(".")[0]
    log = import_event_log(dataset_folder, log_filename)
    log = filter_complete_events(log)

    windows, borders, window_information = window_event_log(log, n_windows=n_windows)

    all_vectors = []
    measure_names = None
    for window_log in windows:
        vector, names = compute_complexity_vector(window_log)
        all_vectors.append(vector)
        if measure_names is None:
            measure_names = names

    complexity_vectors = np.array(all_vectors)

    # If output_subfolder is given (used by compute_log_complexity_sweep.py), all outputs
    # are written to dataset_folder/output_subfolder/ instead of dataset_folder/.
    out_folder = f"{dataset_folder}/{output_subfolder}" if output_subfolder else dataset_folder
    os.makedirs(out_folder, exist_ok=True)  # create the output folder if needed

    # human-readable label for the heatmap only (number + dataset + log name)
    label = f"{log_number:03d}_{dataset_folder}_{log_name}"
    plot_complexity_vectors(complexity_vectors, measure_names, output_path=f"{out_folder}/{label}_heatmap.png",
                            title=label)

    # .npy/.json file names are based on log_name; the analysis notebooks rely on this naming.
    np.save(f"{out_folder}/{log_name}_complexity_vectors.npy", complexity_vectors)
    with open(f"{out_folder}/{log_name}_window_meta.json", "w") as f:
        json.dump(window_information, f)
    with open(f"{out_folder}/{log_name}_borders.json", "w") as f:
        json.dump(borders, f)
    # measure names are saved as well (used by the analysis notebooks)
    with open(f"{out_folder}/{log_name}_measure_names.json", "w") as f:
        json.dump(measure_names, f)

    return len(all_vectors)


if __name__ == "__main__":
    # goal: process every log in every dataset automatically, without changing this by hand
    dataset_folders = ["Bose", "Ceravolo", "Ostovar"]

    # Each dataset uses its own window count (n_windows=10 was too coarse).
    dataset_n_windows = {
        "Bose": 75,
        "Ceravolo": 100,
        "Ostovar": 75,
    }

    next_log_number = 1  # global counter across ALL datasets, 1 to 301
    for dataset_folder in dataset_folders:
        xes_files = [f for f in os.listdir(dataset_folder) if f.endswith(".xes.gz")]
        n_windows_for_this_dataset = dataset_n_windows[dataset_folder]  # window count for this dataset
        print(f"\n Dataset: {dataset_folder} -- {len(xes_files)} files -- n_windows={n_windows_for_this_dataset}")

        for file_idx, log_filename in enumerate(xes_files):
            print(
                f"\n=== Processing log {file_idx + 1}/{len(xes_files)}: {log_filename} (overall #{next_log_number}) ===")
            try:
                n_windows_done = process_single_log(dataset_folder, log_filename, next_log_number, n_windows=n_windows_for_this_dataset)
                print(f"  Saved outputs ({n_windows_done} windows)")
            except Exception as e:
                print(f"  ERROR processing {log_filename}: {e}")
            finally:
                next_log_number += 1  # always increment, even if this log failed

    print("\nDone processing all datasets.")