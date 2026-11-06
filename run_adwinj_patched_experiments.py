
import os
import re
import glob
import pandas as pd
import run_adwinj_experiments as rad


# Settings

SOURCE_DIR = "baselines_results/sweep_adwinj"
OUT_ROOT = "baselines_results/patched_adwinj"

MAX_F1_THRESHOLD = 0.8

COMPLEXITY_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45]

MODE = "combined"

CHANGE_PATTERNS = ["PL", "SM", "CD", "SW"]

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
            f"No measure CSVs found in {SOURCE_DIR}"
        )

    return available


# Parse measures

def parse_measures(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value.lower() in ("", "nan", "none", "[]"):
        return []

    return [
        m.strip()
        for m in value.split(";")
        if m.strip()
    ]



def load_measure_groups(path, f1_threshold):

    df = pd.read_csv(path)

    df["Change Pattern"] = (
        df["Change Pattern"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    groups = {}

    for pattern in CHANGE_PATTERNS:

        row = df[df["Change Pattern"] == pattern]

        if row.empty:
            groups[pattern] = []
        else:
            groups[pattern] = parse_measures(
                row.iloc[0]["Measures"]
            )

        if groups[pattern]:
            print(
                f"[SELECTED] F1 >= {f1_threshold}, "
                f"{pattern}: {len(groups[pattern])} measures"
            )
        else:
            print(
                f"[EMPTY] F1 >= {f1_threshold}, "
                f"{pattern}: no measures"
            )

    groups["INTERSECTION"] = sorted(
        set(groups["PL"])
        & set(groups["SM"])
        & set(groups["CD"])
        & set(groups["SW"])
    )

    groups["UNION"] = sorted(
        set(groups["PL"])
        | set(groups["SM"])
        | set(groups["CD"])
        | set(groups["SW"])
    )

    if not groups["INTERSECTION"]:
        print(
            f"[EMPTY] F1 >= {f1_threshold}: "
            "no common measures in INTERSECTION"
        )

    if not groups["UNION"]:
        print(
            f"[EMPTY] F1 >= {f1_threshold}: "
            "UNION contains no measures"
        )

    return groups


def run_experiment(pattern, measures, f1_threshold,
                   complexity_threshold):

    if not measures:
        print(
            f"[SKIP] {pattern}, F1={f1_threshold}: "
            "no measures available",
            flush=True
        )
        return

    f1_label = str(f1_threshold).replace(".", "_")
    ct_label = str(complexity_threshold).replace(".", "_")

    name = (
        f"ADWINJ_combined_{pattern}"
        f"_f1_{f1_label}"
        f"_ct_{ct_label}"
    )

    output_folder = os.path.join(OUT_ROOT, name)

    done_file = os.path.join(
        output_folder, "evaluation_measures.csv"
    )

    detections_file = os.path.join(
        output_folder, "detections.csv"
    )

    if os.path.exists(done_file) and os.path.exists(detections_file):
        print(f"[SKIP] {name}: already completed", flush=True)
        return

    print(f"\n[START] {name}", flush=True)
    print(f"Number of measures: {len(measures)}", flush=True)
    print(f"F1 threshold: {f1_threshold}", flush=True)
    print(
        f"Complexity threshold: {complexity_threshold}",
        flush=True
    )
    print(f"Measures: {measures}", flush=True)

    rad.COMPLEXITY_THRESHOLD = complexity_threshold

    rad.run(
        run_name=name,
        mode=MODE,
        measures=measures,
        out_root=OUT_ROOT,
        evaluate_noisy=rad.EVALUATE_NOISY
    )

    print(f"[DONE] {name}", flush=True)


# Main

if __name__ == "__main__":

    os.makedirs(OUT_ROOT, exist_ok=True)

    available_csvs = find_available_csvs()

    print("\nAvailable F1 thresholds:")
    for threshold, path in available_csvs:
        print(f"  {threshold}: {path}")

    print("\nComplexity thresholds:")
    print(COMPLEXITY_THRESHOLDS)

    total_planned = 0
    failed = []

    for f1_threshold, csv_path in available_csvs:

        print("\n" + "=" * 65)
        print(f"F1 THRESHOLD = {f1_threshold}")
        print("=" * 65)

        groups = load_measure_groups(
            csv_path,
            f1_threshold
        )

        print("\nSelected groups:")

        for pattern, measures in groups.items():
            print(f"{pattern}: {len(measures)} measures")

        for complexity_threshold in COMPLEXITY_THRESHOLDS:

            print("\n" + "-" * 60)
            print(
                f"COMPLEXITY THRESHOLD = {complexity_threshold}"
            )
            print("-" * 60)

            for pattern, measures in groups.items():

                if measures:
                    total_planned += 1

                try:
                    run_experiment(
                        pattern,
                        measures,
                        f1_threshold,
                        complexity_threshold
                    )

                except Exception as e:
                    print(
                        f"[FAILED] {pattern}, "
                        f"F1={f1_threshold}, "
                        f"CT={complexity_threshold}: {e}",
                        flush=True
                    )

                    failed.append(
                        (pattern, f1_threshold, complexity_threshold)
                    )

    print("\nAll experiments processed.")

    if failed:
        print("\nFailed experiments:")
        for pattern, f1, ct in failed:
            print(f"  {pattern}, F1={f1}, CT={ct}")

