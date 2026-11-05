# Small script to run one experiment per complexity measure with run_approach.run().
# Settings (PEN, N_WINDOWS, LAG_WINDOW, DATASETS_PATH, OUT_ROOT) are imported from run_experiments.py,
# so edit them there. Then run from the repo root:  python run_all_measures.py
#
# With Docker (from the repo root, in PowerShell):
#   docker build -t cbcdd-image .                                                       # once
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_all_measures.py
#
# Each measure is run on its own and saved to runs/<PREFIX>_<measure>/ (characters other than
# letters, digits, '_' and '-' in the measure name are replaced by '_').
#
# With EVALUATE_NOISY = True, run() also saves versatility_per_change_pattern_by_noise.csv.
# Measures whose result files already exist (including that one if EVALUATE_NOISY is on) are
# skipped, so an interrupted run can simply be started again. Measures that raise an error are skipped and listed at the end.
#
# Set MEASURES_TO_RUN to a short list for a quick test.
#
# Afterwards, run combine_versatility.py to merge the per-measure results into one CSV.


import os
import re

from run_approach import run
from run_experiments import ALL_MEASURES, PEN, N_WINDOWS, LAG_WINDOW, DATASETS_PATH, OUT_ROOT

MEASURES_TO_RUN = ["num_event_types", "num_events"]  # or e.g. ["num_event_types", "num_events"] for a test
PREFIX = "run"
EVALUATE_NOISY = False  # if True, run() also saves versatility_per_change_pattern_by_noise.csv


def run_name_for(measure):
    return f"{PREFIX}_" + re.sub(r"[^A-Za-z0-9_\-]", "_", measure)


def is_done(name):
    needed = ["versatility_per_change_pattern.csv", "evaluation_measures.csv", "radar_chart_precision.png", "radar_chart_recall.png", "radar_chart_f1.png"]
    if EVALUATE_NOISY:
        needed += ["versatility_per_change_pattern_by_noise.csv", "evaluation_measures_by_noise.csv"]
    return all(os.path.exists(f"{OUT_ROOT}/{name}/{f}") for f in needed)

if __name__ == "__main__":
    failed = []
    for i, measure in enumerate(MEASURES_TO_RUN, 1):
        name = run_name_for(measure)
        done = is_done(name)
        print(f"[{i}/{len(MEASURES_TO_RUN)}] {measure}" + (" (already done)" if done else ""))
        if done:
            continue
        try:
            run(
                name,
                measures=[measure],
                pen=PEN,
                n_windows=N_WINDOWS,
                lag_window=LAG_WINDOW,
                datasets_path=DATASETS_PATH,
                out_root=OUT_ROOT,
                evaluate_noisy=EVALUATE_NOISY,
            )
        except Exception as e:
            print(f"FAILED {measure}: {e}")
            failed.append(measure)
    if failed:
        print("Failed measures:", failed)