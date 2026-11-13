# Checks how often the per-pattern patch improves exactly the patterns it targets.
# Run from the repo root:  python compile_patch_table.py
#
# Question: for the change patterns that the standard run does not detect perfectly (F1 < 1) and for which
# complementary measures exist, in how many does the per-pattern patched run improve the F1 of that pattern?
#
# Per algorithm it reads:
#   - the standard run (baseline): F1 per change pattern
#   - selected_measures.json in the patched folder (written by the patched scripts): the per-pattern groups. A pattern has
#     complementary measures if it has a group, i.e. at least one measure whose own F1 beats the baseline F1 for that
#     pattern. INTERSECTION and UNION are not part of the per-pattern strategy and are ignored.
#   - the patched runs <PREFIX>_combined_<PATTERN>_alpha_<ALPHA>/versatility_per_change_pattern.csv: F1 of the target
#     pattern when its group is added with weight alpha
# A pattern counts as improved when the patched F1 of that pattern is higher than the baseline F1 of that pattern.
# An undefined (NaN) baseline F1 is treated as 0, as in the selection scripts, so these patterns are kept.
# Runs that are not finished count as missing.
#
# Output in OUT_DIR: patch_summary.csv (one row per algorithm), patch_detail.csv (one row per algorithm and pattern).

import json
import os
import re

import numpy as np
import pandas as pd

ROOT = "baselines_results"
OUT_DIR = f"{ROOT}/comparison"
ALPHAS = [0.5, 0.3, 0.1, 0.05]
ALGORITHMS = {  
    "Bose J": dict(baseline=f"{ROOT}/bose_standard/versatility_per_change_pattern.csv",
                   patched=f"{ROOT}/patched_bose", prefix="BOSE"),
    "Bose WC": dict(baseline=f"{ROOT}/bose_wc_standard/versatility_per_change_pattern.csv",
                    patched=f"{ROOT}/patched_bose_wc", prefix="BOSE_WC"),
    "ADWIN J": dict(baseline=f"{ROOT}/adwin_standard/versatility_per_change_pattern.csv",
                    patched=f"{ROOT}/patched_adwinj", prefix="ADWINJ"),
    "EMD": dict(baseline=f"{ROOT}/emd_standard/versatility_per_change_pattern.csv",
                patched=f"{ROOT}/patched_emd", prefix="EMD"),
    "VC": dict(baseline=f"{ROOT}/vc_standard/versatility_per_change_pattern.csv",
               patched=f"{ROOT}/patched_vc", prefix="VC"),           
}


def load_f1(path):
    df = pd.read_csv(path)
    df["Change Pattern"] = df["Change Pattern"].astype(str).str.strip().str.upper()
    return pd.to_numeric(df.set_index("Change Pattern")["F1-Score"], errors="coerce")


def patched_f1(cfg, pattern, alpha):
    name = re.sub(r"[^A-Za-z0-9_-]", "_", pattern)
    path = f"{cfg['patched']}/{cfg['prefix']}_combined_{name}_alpha_{str(alpha).replace('.', '_')}/versatility_per_change_pattern.csv"
    if not os.path.exists(path):
        return np.nan
    return load_f1(path).get(pattern, np.nan)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    detail_rows, summary_rows = [], []
    for name, cfg in ALGORITHMS.items():
        if not os.path.exists(cfg["baseline"]) or not os.path.exists(f"{cfg['patched']}/selected_measures.json"):
            print(f"[SKIP] {name}: baseline file or selected_measures.json not found")
            continue
        baseline = load_f1(cfg["baseline"])
        groups = json.load(open(f"{cfg['patched']}/selected_measures.json"))
        rows = []
        for pattern, base_raw in baseline.items():
            base = 0.0 if np.isnan(base_raw) else base_raw  # same rule as the selection: undefined baseline = 0
            row = {"Algorithm": name, "Pattern": pattern, "Baseline F1": base_raw, "Not perfect": base < 1,
                   "Complementary measures": 0 if pattern in ("INTERSECTION", "UNION") else len(groups.get(pattern, []))}
            improved, missing = [], 0
            for alpha in ALPHAS:
                f1 = patched_f1(cfg, pattern, alpha)
                row[f"F1 alpha={alpha}"] = f1
                row[f"Improved alpha={alpha}"] = bool(f1 > base) if not np.isnan(f1) else False
                improved.append(row[f"Improved alpha={alpha}"])
                missing += int(np.isnan(f1) and row["Complementary measures"] > 0)
            row["Improved any alpha"] = any(improved)
            row["Runs missing"] = missing
            rows.append(row)
        detail_rows += rows
        target = [r for r in rows if r["Not perfect"] and r["Complementary measures"] > 0]
        summary = {"Algorithm": name, "Patterns": len(rows), "Not perfect (F1<1)": sum(r["Not perfect"] for r in rows),
                   "With complementary measures": len(target)}
        for alpha in ALPHAS:
            summary[f"Improved at alpha={alpha}"] = sum(r[f"Improved alpha={alpha}"] for r in target)
        summary["Improved at any alpha"] = sum(r["Improved any alpha"] for r in target)
        summary["Runs missing"] = sum(r["Runs missing"] for r in target)
        summary_rows.append(summary)

    detail, summary = pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)
    detail.to_csv(f"{OUT_DIR}/patch_detail.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/patch_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {OUT_DIR}/patch_summary.csv and {OUT_DIR}/patch_detail.csv")


if __name__ == "__main__":
    main()