# Small script to merge the versatility results of single-measure runs into one CSV.
# Run it from the repo root after run_all_measures.py:  python combine_versatility.py
#
# With Docker (from the repo root, in PowerShell):
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python combine_versatility.py
#
# Input:    runs/*/config.json and runs/*/versatility_per_change_pattern.csv
#           Only runs with exactly one measure in config.json are used, the measure name is taken from there (not from the folder name).
# Output:   runs/versatility_all_measures.csv
#           One row per measure, one column per change pattern, each cell is the recall value.
#           If several folders contain the same measure, the last one (alphabetically) is kept.
#           runs/versatility_heatmap.png
#           Heatmap of the same table, rows sorted by mean recall (best measure on top),
#           color scale fixed to 0 to 1 so different batches can be compared.

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT_ROOT = "runs"
OUT_PATH = f"{OUT_ROOT}/versatility_all_measures.csv"
HEATMAP_PATH = f"{OUT_ROOT}/versatility_heatmap.png"


def collect(out_root):
    rows = {}
    for config_path in sorted(Path(out_root).glob("*/config.json")):
        run_dir = config_path.parent
        csv_path = run_dir / "versatility_per_change_pattern.csv"
        if not csv_path.exists():
            continue
        measures = json.loads(config_path.read_text())["measures"]
        if len(measures) != 1:
            continue
        df = pd.read_csv(csv_path)
        rows[measures[0]] = df.set_index("Change Pattern")["Recall"]
    combined = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    combined.index.name = "Measure"
    return combined



def plot_heatmap(combined, path):
    data = combined.loc[combined.mean(axis=1).sort_values(ascending=False).index]
    n_rows, n_cols = data.shape
 
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
 
    means = data.mean(axis=1)
 
    fig, (ax, ax_mean) = plt.subplots(
        1, 2, sharey=True,
        figsize=(max(6, 0.9 * n_cols + 5), max(3, 0.3 * n_rows + 1.5)),
        gridspec_kw={"width_ratios": [n_cols, 1.2], "wspace": 0.05},
    )
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_mean.imshow(means.values.reshape(-1, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
 
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel("Change pattern")
    ax.set_ylabel("Complexity measure")
 
    ax_mean.set_xticks([0])
    ax_mean.set_xticklabels(["Mean"], rotation=45, ha="right")
    ax_mean.tick_params(axis="y", left=False)
 
    for i in range(n_rows):
        for j in range(n_cols):
            value = data.iat[i, j]
            if pd.isna(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if value < 0.6 else "black")
        mean = means.iat[i]
        if not pd.isna(mean):
            ax_mean.text(0, i, f"{mean:.2f}", ha="center", va="center", fontsize=6,
                         fontweight="bold", color="white" if mean < 0.6 else "black")
 
    fig.colorbar(im, ax=[ax, ax_mean], label="Recall")
    fig.suptitle("Versatility per change pattern (sorted by mean recall)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
 
 
if __name__ == "__main__":
    combined = collect(OUT_ROOT)
    combined.to_csv(OUT_PATH)
    print(f"Saved {OUT_PATH}: {combined.shape[0]} measures x {combined.shape[1]} change patterns")
    if not combined.empty:
        plot_heatmap(combined, HEATMAP_PATH)
        print(f"Saved {HEATMAP_PATH}")
  