
# Run patched EMD + complexity experiments for CRE, PRE and CF.
# Falls back to lower F1 thresholds when a pattern has no measures.
# Runs multiple alpha values and skips already completed experiments.
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_emd_patched_experiments.py

import os
import re
import glob
import pandas as pd
import time
import run_emd_experiments as rem


# Settings
ALPHAS = [0.5, 0.3, 0.1, 0.05]
FUSION = "weighted_concat"
MAX_F1_THRESHOLD = 0.8

SOURCE_DIR = "baselines_results/sweep_emd"
OUT_ROOT = "baselines_results/patched_emd"

CHANGE_PATTERNS = ["CF", "PRE", "CRE"]


def parse_measures(value):
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
    files = glob.glob(
        os.path.join(SOURCE_DIR, "measures_above_f1_*.csv")
    )

    available = []

    for path in files:
        filename = os.path.basename(path)
        match = re.search(
            r"measures_above_f1_(\d+(?:\.\d+)?)\.csv",
            filename
        )

        if match:
            threshold = float(match.group(1))

            if threshold <= MAX_F1_THRESHOLD:
                available.append((threshold, path))

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

    groups["INTERSECTION"] = sorted(
        set(groups["CRE"])
        & set(groups["PRE"])
        & set(groups["CF"])
    )

    groups["UNION"] = sorted(
        set(groups["CRE"])
        | set(groups["PRE"])
        | set(groups["CF"])
    )

    selected_thresholds["INTERSECTION"] = "combined groups"
    selected_thresholds["UNION"] = "combined groups"

    if not groups["INTERSECTION"]:
        print("\n[EMPTY] INTERSECTION: no common measures.")

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
    name = f"EMD_combined_{pattern}_alpha_{alpha_label}"

    output_folder = os.path.join(OUT_ROOT, name)

    done_file = os.path.join(
        output_folder, "evaluation_measures.csv"
    )
    detections_file = os.path.join(
        output_folder, "detections.csv"
    )

    # Preserve completed experiments from the previous run.
    if os.path.exists(done_file) and os.path.exists(detections_file):
        print(
            f"[SKIP] {name}: already completed.",
            flush=True
        )
        return

    print(f"\n[START] {name}", flush=True)
    print(f"Number of measures: {len(measures)}", flush=True)
    print(f"Alpha: {alpha}", flush=True)
    print(f"Measures: {measures}", flush=True)

    start = time.perf_counter()

    rem.run(
        name,
        feature_mode="combined",
        fusion=FUSION,
        alpha=alpha,
        measures=measures,
        window_size=rem.WINDOW_SIZE,
        step_size=rem.STEP_SIZE,
        lag_window=rem.LAG_WINDOW,
        datasets_path=rem.DATASETS_PATH,
        out_root=OUT_ROOT,
        evaluate_noisy=rem.EVALUATE_NOISY
    )

    elapsed = time.perf_counter() - start

    print(f"[DONE] {name}", flush=True)
    print(f"Time taken: {elapsed / 60:.2f} minutes", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    groups, selected_thresholds = load_measure_groups()

    print("\n" + "=" * 60)
    print("FINAL SELECTED MEASURE GROUPS")
    print("=" * 60)

    for pattern, measures in groups.items():
        print(f"\n{pattern}: {len(measures)} measures")

        if pattern in CHANGE_PATTERNS:
            print(f"Selected threshold: {selected_thresholds[pattern]}")

        if measures:
            print(measures)
        else:
            print("NO MEASURES")

    print(f"\nAlpha values: {ALPHAS}")
    print(
        f"Maximum possible experiments: "
        f"{len(groups) * len(ALPHAS)}"
    )

    failed = []

    for alpha in ALPHAS:
        print(f"\n{'=' * 60}")
        print(f"RUNNING EXPERIMENTS WITH ALPHA = {alpha}")
        print(f"{'=' * 60}")

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
