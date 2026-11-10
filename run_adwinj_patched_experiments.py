import os
import re
import json
import pandas as pd

import run_adwinj_experiments as rad
import run_measure_sweep as rms



BASELINE_FILE = (
    "baselines_results/adwin_standard/versatility_per_change_pattern.csv"
)

SOURCE_DIR = "baselines_results/sweep_adwinj"
OUT_ROOT = "baselines_results/patched_adwinj_new"

MODE = "combined"
ALPHAS = [0.5, 0.3, 0.1, 0.05]

EVALUATE_NOISY = False

# Use the same list of complexity measures as the sweep script.
ALL_MEASURES = rms.ALL_MEASURES



def sanitize_name(name):
    """Convert a measure/pattern name into a folder-safe name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))


def normalize_pattern(pattern):
    """Normalize change-pattern names for comparison."""
    return str(pattern).strip().upper()


def load_f1_results(csv_path):
    """
    Load F1 scores from a versatility_per_change_pattern.csv file.

    Returns:
        Dictionary: {change_pattern: f1_score}
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

        if pd.notna(f1):
            results[pattern] = float(f1)

    return results



def load_baseline():
    """
    Load standard ADWIN/J F1 scores.

    Returns:
        Dictionary: {change_pattern: baseline_f1}
    """
    baseline = load_f1_results(BASELINE_FILE)

    if not baseline:
        raise RuntimeError(
            f"No baseline F1 results found in {BASELINE_FILE}"
        )

    print("\nStandard ADWIN/J baseline:")
    for pattern, f1 in sorted(baseline.items()):
        print(f"  {pattern}: {f1:.4f}")

    return baseline


# ─── Load complexity measure results ─────────────────────────────────────────

def load_measure_results():
    """
    Load the F1 scores of every complexity measure from the sweep.

    Each measure has its own folder:
        baselines_results/sweep_adwinj/<measure>/

    Returns:
        Dictionary:
            {
                measure_name: {
                    change_pattern: f1_score
                }
            }
    """
    if not os.path.isdir(SOURCE_DIR):
        raise RuntimeError(f"Sweep directory not found: {SOURCE_DIR}")

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
            print(
                f"Skipping unknown measure folder: {folder}"
            )
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



def select_measures(baseline, measure_results):
    """
    For each change pattern, select all measures whose F1 is higher
    than the standard ADWIN/J baseline F1 for that pattern.

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

    # Iterate over patterns found in the baseline CSV.
    # No change patterns are hardcoded.
    for pattern, baseline_f1 in baseline.items():

        improving_measures = []

        for measure, results in measure_results.items():

            measure_f1 = results.get(pattern)

            # A missing F1 score cannot be considered an improvement.
            if measure_f1 is None:
                continue

            improvement = measure_f1 - baseline_f1

            is_improving = measure_f1 > baseline_f1

            comparison_rows.append({
                "Change Pattern": pattern,
                "Measure": measure,
                "Baseline F1": baseline_f1,
                "Measure F1": measure_f1,
                "Improvement": improvement,
                "Selected": is_improving
            })

            if is_improving:
                improving_measures.append(measure)

        # Only create a target group if at least one measure improves
        # the baseline for this particular change pattern.
        if improving_measures:
            groups[pattern] = improving_measures

    comparison_df = pd.DataFrame(comparison_rows)


    # These are calculated only across the dynamically selected patterns.
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



def print_selection(groups, baseline):
    print("PATCHING SELECTION")

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
        print(f"\n{pattern} (baseline F1 = {baseline[pattern]:.4f})")

        for measure in groups[pattern]:
            print(f"  - {measure}")

    print("\nINTERSECTION:")
    for measure in groups.get("INTERSECTION", []):
        print(f"  - {measure}")

    print("\nUNION:")
    for measure in groups.get("UNION", []):
        print(f"  - {measure}")

    



def run_experiment(pattern, measures, alpha):
    """
    Run ADWIN/J combined with the selected complexity measures.
    """

    pattern_label = sanitize_name(pattern)
    alpha_label = str(alpha).replace(".", "_")

    run_name = (
        f"ADWINJ_{MODE}_{pattern_label}_alpha_{alpha_label}"
    )

    output_dir = os.path.join(OUT_ROOT, run_name)

    # Avoid rerunning experiments that have already completed.
    evaluation_file = os.path.join(
        output_dir,
        "evaluation_measures.csv"
    )

    detections_file = os.path.join(
        output_dir,
        "detections.csv"
    )

    if (
        os.path.exists(evaluation_file)
        and os.path.exists(detections_file)
    ):
        print(f"\nSkipping completed experiment: {run_name}")
        return

    print("\n" + "-" * 75)
    print(f"Running: {run_name}")
    print(f"Pattern/group: {pattern}")
    print(f"Alpha: {alpha}")
    print(f"Measures ({len(measures)}):")

    for measure in measures:
        print(f"  - {measure}")

    print("-" * 75)

    # ADWIN/J experiment settings.
    rad.ALPHA = alpha
    rad.EVALUATE_NOISY = EVALUATE_NOISY

    rad.run(
        run_name=run_name,
        mode=MODE,
        measures=measures,
        out_root=OUT_ROOT,
        evaluate_noisy=EVALUATE_NOISY
    )



def main():

    os.makedirs(OUT_ROOT, exist_ok=True)

    # 1. Load standard ADWIN/J results.
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

    # 5. Save the dynamically selected groups.
    groups_path = os.path.join(
        OUT_ROOT,
        "selected_measures.json"
    )

    with open(groups_path, "w") as f:
        json.dump(groups, f, indent=4)

    print(f"Selected groups saved to: {groups_path}")

    # 6. Print the selected measures.
    print_selection(groups, baseline)

    if not groups:
        print("\nNo experiments to run.")
        return

    # 7. Run one experiment for every group and alpha.
    for pattern, measures in groups.items():

        if not measures:
            continue

        for alpha in ALPHAS:

            run_experiment(
                pattern=pattern,
                measures=measures,
                alpha=alpha
            )

    print("\nAll patching experiments completed.")


if __name__ == "__main__":
    main()
