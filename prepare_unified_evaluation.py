# Collects baselines_results/<run_name>/detections.csv of all runs into one CSV and writes the matching
# config (shorter_names, used_parameters, colors) for evaluation_unified_framework.ipynb.
# Run from the repo root:  python prepare_unified_evaluation.py
#
# Outputs (overwritten):
#   evaluation_results/all_detections.csv          -> CSV_PATH of the notebook
#   evaluation_results/unified_framework_config.json

import csv
import json
from pathlib import Path

RESULTS_PATH = Path("baselines_results")
CSV_OUT = Path("evaluation_results/all_detections.csv")
CONFIG_OUT = Path("evaluation_results/unified_framework_config.json")

# Run name prefix -> (short name prefix, columns grouped by to find the best parameter setting).
# Runs that don't exist in baselines_results/ are simply not picked up.
# Cluster runs: "Window Size" is left out because it differs per dataset within one run, so grouping by it
# would split the logs of a run. EMD runs: one window size per run, and they have no "Threshold" column.
# Bose and ADWIN runs: one setting per run and no parameter columns, so they are grouped by "Algorithm".
APPROACHES = {
    "VC_": ("VC_", ["Threshold"]),
    "emd_": ("EMD_", ["Window Size"]),
    "bose_": ("Bose_", ["Algorithm"]),
    "adwin_": ("ADWIN_", ["Algorithm"]),
}

COLORS = ["#DB444B", "#9A607F", "#006BA2", "#3EBCD2", "#379A8B", "#EBB434", "#B4BA39", "#D1B07C", "#8E44AD", "#FF7F50"]


def approach_of(algorithm):
    """Run name prefix of the algorithm, e.g. "emd_" for emd_combined_1."""
    for prefix in APPROACHES:
        if algorithm.startswith(prefix):
            return prefix
    raise ValueError(f"Unknown approach of run {algorithm!r}, expected a prefix in {list(APPROACHES)}")


def short_name(algorithm):
    """e.g. cluster_similarity -> VC_similarity, cluster_combined_all_weighted -> VC_combined_AllW,
    emd_standard -> EMD_standard."""
    prefix = approach_of(algorithm)
    name = algorithm.replace("_weighted", "W").replace("_all", "_All")
    return APPROACHES[prefix][0] + name.removeprefix(prefix)


def main():
    rows, fieldnames = [], []
    for path in sorted(RESULTS_PATH.glob("*/detections.csv")):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames += [name for name in reader.fieldnames if name not in fieldnames]
            rows += list(reader)
        print(f"Read {path}")

    algorithms = sorted({row["Algorithm"] for row in rows})
    shorter_names = {algorithm: short_name(algorithm) for algorithm in algorithms}
    if len(set(shorter_names.values())) < len(shorter_names):
        # Initials collide -> keep the full run names
        shorter_names = {algorithm: algorithm for algorithm in algorithms}

    config = {
        "shorter_names": shorter_names,
        "used_parameters": {shorter_names[algorithm]: APPROACHES[approach_of(algorithm)][1] for algorithm in algorithms},
        "colors": [COLORS[i % len(COLORS)] for i in range(len(algorithms))],
    }

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Wrote {len(rows)} rows of {len(algorithms)} runs to {CSV_OUT} and the config to {CONFIG_OUT}")
    for algorithm, name in shorter_names.items():
        print(f"  {algorithm} -> {name}")


if __name__ == "__main__":
    main()
