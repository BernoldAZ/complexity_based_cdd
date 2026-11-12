
import os
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── Settings ────────────────────────────────────────────────────────────────

ROOT = "baselines_results"
OUT_DIR = os.path.join(ROOT, "comparison")

STANDARD_RUNS = {
    "J-Measure": os.path.join(ROOT, "bose_standard"),
    "Window Count": os.path.join(ROOT, "bose_wc_standard"),
    "ADWIN J": os.path.join(ROOT, "adwin_standard"),
    "EMD": os.path.join(ROOT, "emd_standard"),
    "VC": os.path.join(ROOT, "vc_standard"),
}

COLORS = {
    "J-Measure": "#A64D79",
    "Window Count": "#4472C4",
    "ADWIN J": "#70AD47",
    "EMD": "#E69138",
    "VC": "#8E7CC3",
}

GRID_COLUMNS = 3
OUTPUT_FILE = os.path.join(OUT_DIR, "standard_radar_charts.pdf")


# ─── Radar chart ────────────────────────────────────────────────────────────


def plot_radar(ax, algorithm, folder):
    df = pd.read_csv(
        os.path.join(folder, "versatility_per_change_pattern.csv")
    )

    # Same pattern order used by compare_radar_charts.py
    standard = df.set_index("Change Pattern")["F1-Score"]
    patterns = list(standard.index)

    values = [
        0 if pd.isna(standard.get(pattern, np.nan))
        else standard[pattern]
        for pattern in patterns
    ]

    angles = np.linspace(
        0, 2 * np.pi, len(patterns), endpoint=False
    ).tolist()

    angles += angles[:1]
    values += values[:1]

    color = COLORS[algorithm]

    ax.set_theta_offset(0)
    ax.set_theta_direction(1)  

    ax.plot(
        angles,
        values,
        color=color,
        linewidth=1.8
    )
    ax.fill(angles, values, color=color, alpha=0.32)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(patterns, fontsize=12)

    ax.set_ylim(0, 1)
    ax.set_yticklabels([])

    ax.yaxis.grid(True, color="#BFBFBF", linewidth=0.7)
    ax.xaxis.grid(True, color="#BFBFBF", linewidth=0.7)

    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(0.8)

    ax.set_title(
        algorithm,
        fontsize=13,
        fontweight="bold",
        pad=20
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def plot_standard_radars():

    os.makedirs(OUT_DIR, exist_ok=True)

    available_runs = []

    for algorithm, folder in STANDARD_RUNS.items():

        csv_path = os.path.join(
            folder,
            "versatility_per_change_pattern.csv"
        )

        if not os.path.exists(csv_path):
            print(f"[SKIP] {algorithm}: missing {csv_path}")
            continue

        available_runs.append((algorithm, folder))

    if not available_runs:
        print("No standard evaluation files found.")
        return

    n = len(available_runs)
    nrows = math.ceil(n / GRID_COLUMNS)

    fig, axes = plt.subplots(
        nrows,
        GRID_COLUMNS,
        figsize=(7 * GRID_COLUMNS, 7 * nrows),
        subplot_kw={"polar": True},
        squeeze=False
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, (algorithm, folder) in zip(
        axes.ravel(),
        available_runs
    ):
        ax.set_visible(True)
        plot_radar(ax, algorithm, folder)

    fig.suptitle(
        "F1 Scores per Change Pattern – Standard Approaches",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    fig.savefig(
        OUTPUT_FILE,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_standard_radars()
