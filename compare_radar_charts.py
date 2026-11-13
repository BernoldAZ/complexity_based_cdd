# Compares the standard baselines with their patched runs (complexity measures added with a weight alpha).
# Run from the repo root:  python compare_radar_charts.py
#
# For every algorithm in ALGORITHMS it reads the standard run folder and all subfolders of the patched folder.
# Each run folder needs radar_chart_f1.png, versatility_per_change_pattern.csv (F1 per change pattern) and
# evaluation_measures.csv, which run_approach.evaluate() writes.
#
# Output in OUT_DIR:
#   <algorithm>_radar_grid.png   the existing F1 radar charts of all runs, the standard first
#   <algorithm>_radar_best.png   standard and the best patched run on one radar (NaN is drawn as 0, like the originals)
#   <algorithm>_f1_delta.png     F1 of every patched run minus the standard F1, per change pattern (blue = better)
#   summary.csv                  overall metrics, mean F1 per pattern and the differences to the standard
#
# The best run is the one with the highest mean delta (BEST_BY). Note that choosing the best run
# among many on the same logs is optimistic, report it as the best of the sweep.

import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "baselines_results"
OUT_DIR = f"{ROOT}/comparison"
ALGORITHMS = {"Bose J": (f"{ROOT}/bose_standard", f"{ROOT}/patched_bose"),
    "Bose WC": (f"{ROOT}/bose_wc_standard", f"{ROOT}/patched_bose_wc"),
    "ADWIN J": (f"{ROOT}/adwin_standard", f"{ROOT}/patched_adwinj"),
    "EMD": (f"{ROOT}/emd_standard", f"{ROOT}/patched_emd"),
    "VCDD": (f"{ROOT}/vc_standard", f"{ROOT}/patched_vc"),
    "Bose J all measures": (f"{ROOT}/bose_standard", f"{ROOT}/patched_bose_all_measures"),
    "Bose WC all measures": (f"{ROOT}/bose_wc_standard", f"{ROOT}/patched_bose_wc_all_measures"),
    "ADWIN J all measures": (f"{ROOT}/adwin_standard", f"{ROOT}/patched_adwinj_all_measures"),
    "VCDD all measures": (f"{ROOT}/vc_standard", f"{ROOT}/patched_vc_all_measures"),
    "EMD all measures": (f"{ROOT}/emd_standard", f"{ROOT}/patched_emd_all_measures"),

}
BEST_BY = "Delta mean pattern F1"     # or "F1-Score" (overall F1 of evaluation_measures.csv)
GRID_COLUMNS = 4


def label_of(folder):
    name = os.path.basename(folder)
    m = re.search(r"combined_(?P<group>[A-Za-z]+)(?:_f1_(?P<f1>[\d_]+?))?_alpha_(?P<alpha>[\d_]+)$", name)
    if not m:
        return name
    text = f"{m['group']}, alpha={m['alpha'].replace('_', '.')}"
    if m["f1"]:
        text += f", F1>={m['f1'].replace('_', '.')}"
    return text


def collect(standard_dir, patched_root):
    """Collects the standard run and all patched runs that have radar_chart_f1.png."""
    runs = [("standard", standard_dir)]
    for folder in sorted(glob.glob(os.path.join(patched_root, "*"))):
        if os.path.exists(os.path.join(folder, "radar_chart_f1.png")):
            runs.append((label_of(folder), folder))
    return runs


def f1_per_pattern(folder):
    """Reads the F1 per change pattern from versatility_per_change_pattern.csv and returns a Series indexed by change pattern."""
    df = pd.read_csv(os.path.join(folder, "versatility_per_change_pattern.csv"))
    return df.set_index("Change Pattern")["F1-Score"]


def overall_metrics(folder):
    """Read all metrics from evaluation_measures.csv."""
    df = pd.read_csv(os.path.join(folder, "evaluation_measures.csv"))
    row = df.iloc[0].to_dict()

    # Keep the comparison algorithm name separate from the
    # algorithm name stored in evaluation_measures.csv.
    if "Algorithm" in row:
        row["Evaluation Algorithm"] = row.pop("Algorithm")

    return row
def get_run_group(folder):
    """Identify whether a patched run is UNION, INTERSECTION, or another group."""
    name = os.path.basename(folder).upper()

    if re.search(r"(^|_)UNION(_|$)", name):
        return "UNION"

    if re.search(r"(^|_)INTERSECTION(_|$)", name):
        return "INTERSECTION"

    return None


def select_best_run(table, group=None):
    """
    Select the run with the highest Delta mean pattern F1.
    If group is provided, only consider UNION or INTERSECTION runs.
    """
    candidates = table[table["Run"] != "standard"].copy()

    if group is not None:
        candidates = candidates[
            candidates["Folder"].apply(
                lambda folder: get_run_group(folder) == group
            )
        ]

    candidates = candidates.dropna(subset=["Delta mean pattern F1"])

    if candidates.empty:
        return None

    return candidates.loc[candidates["Delta mean pattern F1"].idxmax()]

def plot_grid(algorithm, runs, path):
    """Plots the radar charts of all runs in a grid, the standard first."""
    nrows = math.ceil(len(runs) / GRID_COLUMNS)
    fig, axes = plt.subplots(nrows, GRID_COLUMNS, figsize=(4.5 * GRID_COLUMNS, 4.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, folder) in zip(axes.ravel(), runs):
        ax.imshow(mpimg.imread(os.path.join(folder, "radar_chart_f1.png")))
        ax.set_title(label, fontsize=10, fontweight="bold" if label == "standard" else "normal")
    fig.suptitle(f"{algorithm}: F1 per change pattern", fontsize=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar(algorithm, standard, best_label, best, path, title=None):
    """Plots a radar chart of the standard run and the best patched run."""
    patterns = list(standard.index)
    angles = np.linspace(0, 2 * np.pi, len(patterns), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    for label, series, color in [("standard", standard, "#7F7F7F"), (best_label, best, "#DB444B")]:
        values = [0 if np.isnan(series.get(p, np.nan)) else series[p] for p in patterns]
        values += values[:1]
        is_standard = label == "standard"
        ax.plot(angles, values, color=color, linewidth=2.5, linestyle=":" if is_standard else "-",
            label=label, zorder=3 if is_standard else 2)
        ax.fill(angles, values, color=color, alpha=0.2, zorder=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(patterns, fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title(title or f"{algorithm}: standard vs best patched run", va="bottom", fontsize=16, pad=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=26)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta(algorithm, labels, deltas, path):
    """Plots a heatmap of F1 of every patched run minus the standard F1, per change pattern."""
    data = pd.DataFrame(deltas, index=labels)
    data["mean"] = data.mean(axis=1, skipna=True)
    limit = max(0.05, float(np.nanmax(np.abs(data.values))))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * data.shape[1] + 4), max(3, 0.35 * len(data) + 1.5)))
    norm = plt.Normalize(-limit, limit)
    im = ax.imshow(data.values.astype(float), cmap=cmap, norm=norm, aspect="auto")
    ax.axvline(data.shape[1] - 1.5, color="white", linewidth=3)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iat[i, j]
            if not np.isnan(value):
                r, g, b, _ = cmap(norm(value))
                color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="F1 patched minus standard")
    ax.set_title(f"{algorithm}: change in F1 per change pattern")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for algorithm, (standard_dir, patched_root) in ALGORITHMS.items():
        key = algorithm.replace(" ", "_")
        if not os.path.exists(os.path.join(standard_dir, "radar_chart_f1.png")):
            print(f"[SKIP] {algorithm}: no radar_chart_f1.png in {standard_dir}")
            continue
        runs = collect(standard_dir, patched_root)
        print(f"{algorithm}: standard + {len(runs) - 1} patched runs")
        plot_grid(algorithm, runs, f"{OUT_DIR}/{key}_radar_grid.png")
        if len(runs) == 1:
            continue

        f1 = {label: f1_per_pattern(folder) for label, folder in runs}
        standard = f1["standard"]
        mean_deltas = {}

        for label, folder in runs:
            delta = f1[label] - standard

            # Same calculation used for the heatmap's mean column.
            mean_delta = np.nanmean([
                f1[label].get(pattern, np.nan) - standard.get(pattern, np.nan)
                for pattern in standard.index
            ])

            mean_deltas[label] = mean_delta

            rows.append({
                "Algorithm": algorithm,
                "Run": label,
                "Folder": os.path.basename(folder),
                **overall_metrics(folder),
                "Mean pattern F1": f1[label].mean(),
                "Delta mean pattern F1": mean_delta,
                "Patterns improved": int((delta > 0).sum()),
                "Patterns worse": int((delta < 0).sum()),
            })
        table = pd.DataFrame([r for r in rows if r["Algorithm"] == algorithm])

        # ─── Best overall configuration ────────────────────────────────────────

        best = select_best_run(table)

        if best is not None:
            print(
                f"  best overall: {best['Run']} "
                f"(mean delta: {best['Delta mean pattern F1']:+.3f})"
            )

            plot_radar(
                algorithm,
                standard,
                best["Run"],
                f1[best["Run"]],
                f"{OUT_DIR}/{key}_radar_best.png"
            )


        # ─── Best UNION configuration ──────────────────────────────────────────

        best_union = select_best_run(table, group="UNION")

        if best_union is not None:
            print(
                f"  best UNION: {best_union['Run']} "
                f"(mean delta: {best_union['Delta mean pattern F1']:+.3f})"
            )

            plot_radar(
                algorithm,
                standard,
                best_union["Run"],
                f1[best_union["Run"]],
                f"{OUT_DIR}/{key}_radar_best_union.png",
                title=f"{algorithm}: standard vs best UNION"
            )
        else:
            print("  no UNION runs found")


        # ─── Best INTERSECTION configuration ──────────────────────────────────

        best_intersection = select_best_run(table, group="INTERSECTION")

        if best_intersection is not None:
            print(
                f"  best INTERSECTION: {best_intersection['Run']} "
                f"(mean delta: {best_intersection['Delta mean pattern F1']:+.3f})"
            )

            plot_radar(
                algorithm,
                standard,
                best_intersection["Run"],
                f1[best_intersection["Run"]],
                f"{OUT_DIR}/{key}_radar_best_intersection.png",
                title=f"{algorithm}: standard vs best INTERSECTION"
            )
        else:
            print("  no INTERSECTION runs found")        
        labels = [label for label, _ in runs[1:]]
        plot_delta(algorithm, labels, {p: [f1[l].get(p, np.nan) - standard.get(p, np.nan) for l in labels]
                                       for p in standard.index}, f"{OUT_DIR}/{key}_f1_delta.png")

    if rows:
        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/summary.csv", index=False)
        print(f"Saved {OUT_DIR}/summary.csv")


if __name__ == "__main__":
    main()