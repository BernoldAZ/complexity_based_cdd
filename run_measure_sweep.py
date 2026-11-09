# Small script to run the baselines in "complexity" feature mode, once per complexity measure, and to
# summarize the results of each approach once its runs are done.
# The settings of each baseline (PEN, N_WINDOWS, WINDOW_SIZE, ...) are imported from run_cluster_experiments.py,
# run_emd_experiments.py, run_bose_experiments.py and run_adwinj_experiments.py, so edit them there (their MODE and
# MEASURES are ignored here). Then run from the repo root:  python run_measure_sweep.py
#
# With Docker (from the repo root, in PowerShell):
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_measure_sweep.py
#
# Each (approach, measure) is saved to baselines_results/sweep_<approach>/<measure>/ (characters other than
# letters, digits, '_' and '-' in the measure name are replaced by '_'). Runs whose
# versatility_per_change_pattern.csv already exists are skipped, so an interrupted sweep can simply be started
# again. Runs that raise an error are skipped and listed at the end.
#
# After the runs of an approach, these summaries are written to baselines_results/sweep_<approach>/
# (from all single-measure runs in that folder, see analyze_results.collect_table):
#   - evaluation_all_measures.csv       one row per measure, same columns as evaluation_measures.csv,
#                                       sorted by F1-Score (best first)
#   - f1_per_change_pattern.csv         F1 per measure (rows) and change pattern (columns), noiseless logs
#   - f1_heatmap.png                    that table as a heatmap with a mean column, sorted by mean F1
#   - measures_above_f1_<X>.csv         one row per change pattern: the measures with F1 > F1_THRESHOLD,
#                                       best first
#
# Set MEASURES_TO_RUN to a short list for a quick test. Set ANALYZE_ONLY = True to only redo the summaries.

import os
import re

import pandas as pd

import run_adwinj_experiments as raj
import run_bose_experiments as rbe
import run_cluster_experiments as rce
import run_emd_experiments as rem
from analyze_results import collect_table, heatmap
from constants import ALL_MEASURES

APPROACHES = ["bose_wc"]  # ["vc", "emd", "bose", "adwinj"]  keys of RUNNERS
MEASURES_TO_RUN = ALL_MEASURES  # or e.g. ["num_event_types", "num_events"] for a test
EVALUATE_NOISY = False
F1_THRESHOLD = 0.80
ANALYZE_ONLY = False            # True: skip all runs and only redo the summaries from the existing results
OUT_ROOT = "baselines_results"


def run_vc(name, measure, out_root):
    rce.run(name, feature_mode="complexity", measures=[measure], pen=rce.PEN, n_windows=rce.N_WINDOWS,
            min_cluster_size=rce.MIN_CLUSTER_SIZE, threshold=rce.THRESHOLD, lag_window=rce.LAG_WINDOW,
            datasets_path=rce.DATASETS_PATH, out_root=out_root, evaluate_noisy=EVALUATE_NOISY)


def run_emd(name, measure, out_root):
    rem.run(name, feature_mode="complexity", measures=[measure], window_size=rem.WINDOW_SIZE,
            step_size=rem.STEP_SIZE, lag_window=rem.LAG_WINDOW, datasets_path=rem.DATASETS_PATH,
            out_root=out_root, evaluate_noisy=EVALUATE_NOISY)


def run_bose(name, measure, out_root):
    rbe.run(name, "complexity", [measure], out_root, evaluate_noisy=EVALUATE_NOISY)


def run_adwinj(name, measure, out_root):
    raj.run(name, "complexity", [measure], out_root, evaluate_noisy=EVALUATE_NOISY)


RUNNERS = {"vc": run_vc, "emd": run_emd, "bose": run_bose, "bose_wc": run_bose, "adwinj": run_adwinj}


def run_name_for(measure):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", measure)


def measures_above(f1, threshold):
    rows = []
    for cp in f1.columns:
        above = f1[cp][f1[cp] > threshold].sort_values(ascending=False)
        rows.append({"Change Pattern": cp, "Count": len(above), "Measures": "; ".join(above.index)})
    return rows


def analyze(approach, root):
    summary = collect_table(root, "evaluation_measures.csv", ["Measure"])
    if not summary.empty:
        summary.sort_values("F1-Score", ascending=False).to_csv(f"{root}/evaluation_all_measures.csv", index=False)

    versatility = collect_table(root, "versatility_per_change_pattern.csv", ["Measure", "Change Pattern"])
    if versatility.empty:
        print(f"{approach}: no versatility results to summarize")
        return
    f1 = versatility.pivot(index="Measure", columns="Change Pattern", values="F1-Score").sort_index()
    f1.to_csv(f"{root}/f1_per_change_pattern.csv")
    heatmap(f1.loc[f1.mean(axis=1).sort_values(ascending=False).index], f"{root}/f1_heatmap.png",
            f"{approach}: F1 per change pattern (sorted by mean F1)", "Change pattern",
            label="F1-Score", with_mean=True)
    pd.DataFrame(measures_above(f1, F1_THRESHOLD)).to_csv(
        f"{root}/measures_above_f1_{F1_THRESHOLD}.csv", index=False)
    print(f"{approach}: summarized {f1.shape[0]} measures in {root}")


if __name__ == "__main__":
    failed = []
    for approach in APPROACHES:
        out_root = f"{OUT_ROOT}/sweep_{approach}"
        for i, measure in enumerate([] if ANALYZE_ONLY else MEASURES_TO_RUN, 1):
            name = run_name_for(measure)
            done = os.path.exists(f"{out_root}/{name}/versatility_per_change_pattern.csv")
            print(f"[{approach} {i}/{len(MEASURES_TO_RUN)}] {measure}" + (" (already done)" if done else ""))
            if done:
                continue
            try:
                RUNNERS[approach](name, measure, out_root)
            except Exception as e:
                print(f"FAILED {approach} {measure}: {e}")
                failed.append((approach, measure))
        if os.path.isdir(out_root):
            analyze(approach, out_root)
    if failed:
        print("Failed runs:", failed)
