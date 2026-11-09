# Builds a table comparing, per approach and change pattern, the F1 of the standard approach with the number of
# complexity measures that reach F1 > F1_THRESHOLD when run alone in that approach (the sweep_<approach> runs).
# Same strict threshold as the measures_above_f1_<X>.csv files of run_measure_sweep.py. Run from the repo root:
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python build_standard_vs_complexity_table.py
# Output: baselines_results/standard_vs_complexity_f1_<F1_THRESHOLD>.csv

import pandas as pd

F1_THRESHOLD = 0.8
OUT_ROOT = "baselines_results"

# approach name -> (standard run folder, sweep folder)
APPROACHES = {
    "Bose (J)": ("bose_standard", "sweep_bose_j"),
    "ADWIN-J": ("adwin_standard", "sweep_adwinj"),
    "EMD": ("emd_standard", "sweep_emd"),
    "VC": ("vc_standard", "sweep_vc"),
}


def build_table():
    rows = []
    for name, (standard, sweep) in APPROACHES.items():
        standard_f1 = pd.read_csv(f"{OUT_ROOT}/{standard}/versatility_per_change_pattern.csv").set_index(
            "Change Pattern")["F1-Score"]
        sweep_f1 = pd.read_csv(f"{OUT_ROOT}/{sweep}/f1_per_change_pattern.csv").set_index("Measure")
        for pattern in sweep_f1.columns:
            f1 = standard_f1.get(pattern)
            rows.append({
                "Approach": name,
                "Change Pattern": pattern,
                "Standard F1": round(float(f1), 2) if pd.notna(f1) else 0.0,  # no detection -> 0
                f"Complexity measures with F1 > {F1_THRESHOLD}": int((sweep_f1[pattern] > F1_THRESHOLD).sum()),
                "Measures evaluated": len(sweep_f1),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build_table()
    path = f"{OUT_ROOT}/standard_vs_complexity_f1_{F1_THRESHOLD}.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"Written to {path}")
