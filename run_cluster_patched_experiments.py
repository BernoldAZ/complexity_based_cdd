import os
import re
import json
import pandas as pd

import run_cluster_experiments as rvc
import numpy as np


BASELINE_FILE = (
    "baselines_results/vc_standard/versatility_per_change_pattern.csv"
)

SOURCE_DIR = "baselines_results/sweep_vc"

OUT_ROOT = "baselines_results/patched_vc"

FEATURE_MODE = "combined"
FUSION = "weighted_concat"

ALPHAS = [0.5, 0.3, 0.1, 0.05]

EVALUATE_NOISY = False

# Use exactly the measures recognized by run_cluster_experiments.py.
ALL_MEASURES = rvc.ALL_MEASURES

# Use the same VC parameters as the original experiment.
PEN = rvc.PEN
N_WINDOWS = rvc.N_WINDOWS
MIN_CLUSTER_SIZE = rvc.MIN_CLUSTER_SIZE
THRESHOLD = rvc.THRESHOLD
LAG_WINDOW = rvc.LAG_WINDOW
DATASETS_PATH = rvc.DATASETS_PATH



def sanitize_name(name):
    """Convert a name into a folder-safe name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))


def normalize_pattern(pattern):
    """Normalize change-pattern names for comparison."""
    return str(pattern).strip().upper()


def load_f1_results(csv_path):
    """
    Load F1 scores from a versatility_per_change_pattern.csv file.

    Returns:
        {change_pattern: f1_score}
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)

    required_columns = {"Change Pattern", "F1-Score"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {csv_path}. "
            f"Expected {required_columns}, got {set(df.columns)}"
        )

    results = {}

    for _, row in df.iterrows():
        pattern = normalize_pattern(row["Change Pattern"])
        f1 = pd.to_numeric(row["F1-Score"], errors="coerce")

        results[pattern] = float(f1) if pd.notna(f1) else np.nan


    return results



def load_baseline():

    baseline = load_f1_results(BASELINE_FILE)

    if not baseline:
        raise RuntimeError(
            f"No baseline F1 results found in {BASELINE_FILE}"
        )

    print("\nStandard VC baseline:")

    for pattern, f1 in sorted(baseline.items()):
        print(f"  {pattern}: {f1:.4f}")

    return baseline



def load_measure_results():

    if not os.path.isdir(SOURCE_DIR):
        raise RuntimeError(
            f"VC sweep directory not found: {SOURCE_DIR}"
        )

    # Map sanitized folder names back to their original measure names.
    folder_to_measure = {
        sanitize_name(measure): measure
        for measure in ALL_MEASURES
    }

    measure_results = {}

    for folder in sorted(os.listdir(SOURCE_DIR)):

        folder_path = os.path.join(SOURCE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        if folder not in folder_to_measure:
            print(f"Skipping unknown measure folder: {folder}")
            continue

        measure = folder_to_measure[folder]

        csv_path = os.path.join(
            folder_path,
            "versatility_per_change_pattern.csv"
        )

        if not os.path.exists(csv_path):
            print(
                f"Skipping {measure}: versatility CSV not found"
            )
            continue

        results = load_f1_results(csv_path)

        if results:
            measure_results[measure] = results

            print(
                f"Loaded {measure}: {len(results)} change patterns"
            )

    if not measure_results:
        raise RuntimeError(
            f"No complexity measure results found in {SOURCE_DIR}"
        )

    return measure_results


# ─── Select measures that improve the baseline ───────────────────────────────

def select_measures(baseline, measure_results):
    """
    Select measures whose F1 is strictly higher than standard VC
    for each change pattern.

    Change patterns are discovered dynamically from the baseline CSV.

    Returns:
        groups:
            {
                change_pattern: [measure1, measure2, ...],
                "INTERSECTION": [...],
                "UNION": [...]
            }

        comparison_df:
            Detailed comparison of baseline and measure F1 scores.
    """

    groups = {}
    comparison_rows = []

    # Iterate over patterns found in the standard VC results.
    # No change patterns are hardcoded.
    for pattern, baseline_f1 in baseline.items():

        improving_measures = []

        for measure, results in measure_results.items():

            measure_f1 = results.get(pattern)

            # Missing results cannot be considered improvements.
            if measure_f1 is None:
                continue

            # Treat an undefined standard F1 as zero for comparison.
            comparison_baseline_f1 = (
                0.0 if pd.isna(baseline_f1) else baseline_f1
            )

            if pd.isna(measure_f1):
                continue

            improvement = measure_f1 - comparison_baseline_f1

            is_improving = measure_f1 > comparison_baseline_f1

            comparison_rows.append({
                "Change Pattern": pattern,
                "Measure": measure,
                "Baseline F1": baseline_f1,
                "Comparison Baseline F1": comparison_baseline_f1,
                "Measure F1": measure_f1,
                "Improvement": improvement,
                "Selected": is_improving
            })

            if is_improving:
                improving_measures.append(measure)

        # Only target patterns with at least one improving measure.
        if improving_measures:
            groups[pattern] = improving_measures

    comparison_df = pd.DataFrame(comparison_rows)

    # ─── INTERSECTION and UNION ───────────────────────────────────────────

    # Calculate these only across the dynamically selected patterns.
    target_patterns = list(groups.keys())

    if target_patterns:

        measure_sets = [
            set(groups[pattern])
            for pattern in target_patterns
        ]

        intersection = set.intersection(*measure_sets)
        union = set.union(*measure_sets)

        if intersection:
            groups["INTERSECTION"] = sorted(intersection)

        if union:
            groups["UNION"] = sorted(union)

    return groups, comparison_df


# ─── Print selection results ─────────────────────────────────────────────────

def print_selection(groups, baseline):

    print("\n" + "=" * 75)
    print("VC PATCHING SELECTION")
    print("=" * 75)

    target_patterns = [
        pattern
        for pattern in groups
        if pattern not in ("INTERSECTION", "UNION")
    ]

    if not target_patterns:
        print("\nNo complexity measure improves the baseline.")
        return

    print("\nTarget change patterns:")

    for pattern in target_patterns:

        print(
            f"\n{pattern} (baseline F1 = {baseline[pattern]:.4f})"
        )

        for measure in groups[pattern]:
            print(f"  - {measure}")

    print("\nINTERSECTION:")

    for measure in groups.get("INTERSECTION", []):
        print(f"  - {measure}")

    print("\nUNION:")

    for measure in groups.get("UNION", []):
        print(f"  - {measure}")

    print("\n" + "=" * 75)


# ─── Run one VC patching experiment ──────────────────────────────────────────

def run_experiment(pattern, measures, alpha):

    pattern_label = sanitize_name(pattern)
    alpha_label = str(alpha).replace(".", "_")

    run_name = (
        f"VC_{FEATURE_MODE}_{pattern_label}_alpha_{alpha_label}"
    )

    output_dir = os.path.join(OUT_ROOT, run_name)

    evaluation_file = os.path.join(
        output_dir,
        "evaluation_measures.csv"
    )

    detections_file = os.path.join(
        output_dir,
        "detections.csv"
    )

    # Skip only experiments that have already completed.
    if (
        os.path.exists(evaluation_file)
        and os.path.exists(detections_file)
    ):
        print(f"\nSkipping completed experiment: {run_name}")
        return

    print(f"Running: {run_name}")
    print(f"Pattern/group: {pattern}")
    print(f"Alpha: {alpha}")
    print(f"Fusion: {FUSION}")
    print(f"Measures ({len(measures)}):")

    for measure in measures:
        print(f"  - {measure}")


    # Run VC with similarity + selected complexity measures.
    rvc.run(
        run_name=run_name,
        feature_mode=FEATURE_MODE,
        fusion=FUSION,
        alpha=alpha,
        measures=measures,
        pen=PEN,
        n_windows=N_WINDOWS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        threshold=THRESHOLD,
        lag_window=LAG_WINDOW,
        datasets_path=DATASETS_PATH,
        out_root=OUT_ROOT,
        evaluate_noisy=EVALUATE_NOISY
    )



def main():

    os.makedirs(OUT_ROOT, exist_ok=True)

    # 1. Load standard VC results.
    baseline = load_baseline()

    # 2. Load individual complexity measure results.
    measure_results = load_measure_results()

    # 3. Compare each measure against the baseline.
    groups, comparison_df = select_measures(
        baseline,
        measure_results
    )

    # 4. Save the complete comparison.
    comparison_path = os.path.join(
        OUT_ROOT,
        "measure_comparison.csv"
    )

    comparison_df.to_csv(comparison_path, index=False)

    print(f"\nComparison saved to: {comparison_path}")

    # 5. Save the selected groups.
    groups_path = os.path.join(
        OUT_ROOT,
        "selected_measures.json"
    )

    with open(groups_path, "w") as f:
        json.dump(groups, f, indent=4)

    print(f"Selected groups saved to: {groups_path}")

    # 6. Print the selection.
    print_selection(groups, baseline)

    if not groups:
        print("\nNo experiments to run.")
        return

    # 7. Run every selected group for every alpha.
    for pattern, measures in groups.items():

        if not measures:
            continue

        for alpha in ALPHAS:

            run_experiment(
                pattern=pattern,
                measures=measures,
                alpha=alpha
            )

    print("\nAll VC patching experiments completed.")


if __name__ == "__main__":
    main()
