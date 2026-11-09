
import os
import re
import glob
import pandas as pd
import run_bose_experiments as rb

SOURCE_DIR = "baselines_results/sweep_bose_wc" #"sweep_bose_wc" for bose using Windows Count,  "sweep_bose_J" for bose using J Measure
OUT_ROOT = "baselines_results/patched_bose_wc" #"patched_bose_wc" for bose using Windows Count,  "patched_bose" for bose using J Measure

MAX_F1_THRESHOLD = 0.8

MODE = "combined"
ALPHAS = [0.5, 0.3, 0.1, 0.05]
CHANGE_PATTERNS = ["CD", "PL", "SM", "PM"] #  ["CD", "PL", "SM", "PM"] for bose using Windows Count,  ["CD", "PL", "SM"] for bose using J Measure


def parse_measures(value):
    """Convert the Measures CSV cell into a list of metric names."""
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value.lower() in ("", "nan", "none", "[]"):
        return []

    return [
        measure.strip()
        for measure in value.split(";")
        if measure.strip()
    ]


def find_available_csvs():
    """Find all measure CSVs at or below the maximum F1 threshold."""
    files = glob.glob(
        os.path.join(SOURCE_DIR, "measures_above_f1_*.csv")
    )

    available = []

    for path in files:
        filename = os.path.basename(path)
        match = re.search(r"measures_above_f1_(\d+(?:\.\d+)?)\.csv", filename)

        if match:
            threshold = float(match.group(1))

            if threshold <= MAX_F1_THRESHOLD:
                available.append((threshold, path))

    # Highest threshold first.
    available.sort(key=lambda x: x[0], reverse=True)

    if not available:
        raise FileNotFoundError(
            f"No measures_above_f1_*.csv files found in {SOURCE_DIR}"
        )

    return available


def load_measure_groups():
    available_csvs = find_available_csvs()

    print("\nAvailable F1 threshold files:")
    for threshold, path in available_csvs:
        print(f"  F1 >= {threshold}: {path}")

    groups = {}
    selected_thresholds = {}

    for pattern in CHANGE_PATTERNS:
        groups[pattern] = []
        selected_thresholds[pattern] = None

        # Search from the highest available threshold downwards.
        for threshold, path in available_csvs:
            df = pd.read_csv(path)

            df["Change Pattern"] = (
                df["Change Pattern"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            row = df[df["Change Pattern"] == pattern]

            if row.empty:
                continue

            measures = parse_measures(row.iloc[0]["Measures"])

            if measures:
                groups[pattern] = measures
                selected_thresholds[pattern] = threshold

                print(
                    f"\n[SELECTED] {pattern}: "
                    f"{len(measures)} measures from F1 >= {threshold}"
                )
                break

        if not groups[pattern]:
            lowest_threshold = available_csvs[-1][0]

            print(
                f"\n[EMPTY] {pattern}: no measures found, "
                f"even at the lowest available threshold "
                f"(F1 >= {lowest_threshold})."
            )

    # Calculate intersection and union using the selected groups.
    groups["INTERSECTION"] = sorted(set.intersection(*(set(groups[p]) for p in CHANGE_PATTERNS)))


    groups["UNION"] = sorted(set.union(*(set(groups[p]) for p in CHANGE_PATTERNS)))
    

    selected_thresholds["INTERSECTION"] = "combined groups"
    selected_thresholds["UNION"] = "combined groups"

    if not groups["INTERSECTION"]:
        print(
            "\n[EMPTY] INTERSECTION: no measures are common "
            "to all three selected groups."
        )

    if not groups["UNION"]:
        print("\n[EMPTY] UNION: no measures found in any group.")

    return groups, selected_thresholds


def run_experiment(pattern, measures, alpha):
    if not measures:
        print(
            f"[SKIP] {pattern}: no measures available.",
            flush=True
        )
        return

    alpha_label = str(alpha).replace(".", "_")
    name = f"BOSE_WC_combined_{pattern}_alpha_{alpha_label}"


    output_folder = os.path.join(OUT_ROOT, name)
    done_file = os.path.join(output_folder, "evaluation_measures.csv")
    detections_file = os.path.join(output_folder, "detections.csv")

    if os.path.exists(done_file) and os.path.exists(detections_file):
        print(f"[SKIP] {name}: already completed", flush=True)
        return

    print(f"\n[START] {name}", flush=True)
    print(f"Number of measures: {len(measures)}", flush=True)
    print(f"Alpha: {alpha}", flush=True)
    print(f"Measures: {measures}", flush=True)

    rb.ALPHA = alpha
    
    rb.run(
        run_name=name,
        mode=MODE,
        measures=measures,
        out_root=OUT_ROOT,
        evaluate_noisy=rb.EVALUATE_NOISY
    )

    print(f"[DONE] {name}", flush=True)


if __name__ == "__main__":
    assert rb.BOSE_MEASURE == "WC", "set BOSE_MEASURE = 'WC' in run_bose_experiments.py"
    os.makedirs(OUT_ROOT, exist_ok=True)

    groups, selected_thresholds = load_measure_groups()

    print("FINAL SELECTED MEASURE GROUPS")

    for pattern, measures in groups.items():
        print(f"\n{pattern}: {len(measures)} measures")

        if pattern in CHANGE_PATTERNS:
            print(f"Selected threshold: {selected_thresholds[pattern]}")

        if measures:
            print(measures)
        else:
            print("NO MEASURES")

    total_experiments = sum(
        bool(measures) for measures in groups.values()
    ) * len(ALPHAS)

    print(f"\nAlpha values: {ALPHAS}")
    print(f"Maximum possible experiments: {len(groups) * len(ALPHAS)}")
    print(f"Experiments with non-empty groups: {total_experiments}")

    failed = []

    for alpha in ALPHAS:
        print(f"RUNNING EXPERIMENTS WITH ALPHA = {alpha}")

        for pattern, measures in groups.items():
            try:
                run_experiment(pattern, measures, alpha)
            except Exception as e:
                print(
                    f"[FAILED] {pattern}, alpha={alpha}: {e}",
                    flush=True
                )
                failed.append((pattern, alpha))

    print("\nAll experiments processed.")

    if failed:
        print("\nFailed experiments:")
        for pattern, alpha in failed:
            print(f"  {pattern}, alpha={alpha}")
