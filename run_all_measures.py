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
# Measures whose versatility_per_change_pattern.csv already exists are skipped, so an interrupted
# run can simply be started again. Measures that raise an error are skipped and listed at the end.
# Set MEASURES_TO_RUN to a short list for a quick test.
#
# Afterwards, run combine_versatility.py to merge the per-measure results into one CSV.


import os
import re

from run_approach import run
from run_experiments import ALL_MEASURES, PEN, N_WINDOWS, LAG_WINDOW, DATASETS_PATH, OUT_ROOT

MEASURES_TO_RUN = ALL_MEASURES  # or e.g. ["num_event_types", "num_events"] for a test
PREFIX = "run"


def run_name_for(measure):
    return f"{PREFIX}_" + re.sub(r"[^A-Za-z0-9_\-]", "_", measure)


if __name__ == "__main__":
    failed = []
    completed = 0
    skipped = 0
    total = len(MEASURES_TO_RUN)
    print(f"Starting {total} measures")
    for i, measure in enumerate(MEASURES_TO_RUN, 1):
        name = run_name_for(measure)
        done = os.path.exists(f"{OUT_ROOT}/{name}/versatility_per_change_pattern.csv")
        print(f"[{i}/{total}] Checking {measure}")
        if done:
            skipped += 1
            print(f"[{i}/{total}] Skipping {measure}: output already exists ({completed} completed, {skipped} skipped)")
            continue
        try:
            print(f"[{i}/{total}] Running {measure}")
            run(
                name,
                measures=[measure],
                pen=PEN,
                n_windows=N_WINDOWS,
                lag_window=LAG_WINDOW,
                datasets_path=DATASETS_PATH,
                out_root=OUT_ROOT,
            )
            completed += 1
            print(f"[{i}/{total}] Completed {measure} ({completed} completed, {skipped} skipped)")
        except Exception as e:
            print(f"FAILED {measure}: {e}")
            failed.append(measure)
            print(f"[{i}/{total}] Progress: {completed} completed, {skipped} skipped, {len(failed)} failed")
    print(f"Finished: {completed} completed, {skipped} skipped, {len(failed)} failed out of {total}")
    if failed:
        print("Failed measures:", failed)