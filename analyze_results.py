# Small script to merge the results of single-measure runs and plot a quick analysis.
# Run it from the repo root after run_all_measures.py (with EVALUATE_NOISY = True):  python analyze_results.py
#
# With Docker (from the repo root, in PowerShell):
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python analyze_results.py
#
# Input:    runs/*/config.json plus, per run, evaluation_measures.csv, versatility_per_change_pattern.csv,
#           evaluation_measures_by_noise.csv and versatility_per_change_pattern_by_noise.csv.
#           Only runs with exactly one measure in config.json are used, the measure name is taken from
#           there (not from the folder name). If several folders contain the same measure, the last one
#           (alphabetically) is kept. The two by_noise files only exist for runs made with evaluate_noisy=True.
# Output:   the merged tables in runs/ (evaluation_all_measures.csv, versatility_all_measures.csv,
#           evaluation_by_noise_all_measures.csv, versatility_by_noise_all_measures.csv) and in runs/plots/:
#           -    scorecard.png                       measures x F1, precision, recall, latency, versatility, robustness
#           -    versatility_heatmap.png             recall per change pattern on the noiseless logs, with a mean column,
#                                                   rows sorted by mean recall (best measure on top)
#           -    noise_curves.png                    harmonic mean of F1, latency and versatility against noise level,
#                                                   one panel per log source, the top measures highlighted
#           -    noisy_vs_noiseless_<source>.png     recall(noisy) minus recall(noiseless) per measure and change
#                                                   pattern, blue = better with noise, red = worse. n is the number
#                                                   of pooled noisy logs behind each pattern.

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_ROOT = "runs"
PLOT_DIR = f"{OUT_ROOT}/plots"
SCORECARD_COLUMNS = ["F1-Score", "Precision", "Recall", "Latency (Score)", "Versatility", "Robustness"]
HARMONIC_COLUMN = "Harmonic Mean (F1, Latency, Versatility)"
TOP_N = 5


def collect_table(out_root, filename, keys):
    frames = []
    for config_path in sorted(Path(out_root).glob("*/config.json")):
        csv_path = config_path.parent / filename
        if not csv_path.exists():
            continue
        measures = json.loads(config_path.read_text())["measures"]
        if len(measures) != 1:
            continue
        df = pd.read_csv(csv_path, dtype={"Noise Level": str})
        df.insert(0, "Measure", measures[0])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(keys, keep="last")


def heatmap(data, path, title, xlabel, cmap_name="viridis", vmin=0.0, vmax=1.0, label="Score", with_mean=False):
    n_rows, n_cols = data.shape
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgray")
    norm = plt.Normalize(vmin, vmax)
    width = max(6, 0.9 * n_cols + 4)
    height = max(3, 0.3 * n_rows + 1.5)

    if with_mean:
        fig, (ax, ax_mean) = plt.subplots(
            1, 2, sharey=True, figsize=(width + 1, height),
            gridspec_kw={"width_ratios": [n_cols, 1.2], "wspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        ax_mean = None

    def annotate(axis, i, j, value, bold=False):
        if pd.isna(value):
            return
        r, g, b, _ = cmap(norm(value))
        color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
        axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color,
                  fontweight="bold" if bold else "normal")

    im = ax.imshow(data.values.astype(float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Complexity measure")
    for i in range(n_rows):
        for j in range(n_cols):
            annotate(ax, i, j, data.iat[i, j])

    if ax_mean is not None:
        means = data.mean(axis=1)
        ax_mean.imshow(means.values.astype(float).reshape(-1, 1), cmap=cmap, norm=norm, aspect="auto")
        ax_mean.set_xticks([0])
        ax_mean.set_xticklabels(["Mean"], rotation=45, ha="right")
        ax_mean.tick_params(axis="y", left=False)
        for i in range(n_rows):
            annotate(ax_mean, i, 0, means.iat[i], bold=True)

    fig.colorbar(im, ax=[a for a in (ax, ax_mean) if a is not None], label=label)
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(summary, path):
    columns = [c for c in SCORECARD_COLUMNS if c in summary.columns]
    data = summary.set_index("Measure")[columns]
    data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
    heatmap(data, path, "Overall results (sorted by mean of the columns)", "")


def plot_versatility(versatility, path):
    data = versatility.pivot(index="Measure", columns="Change Pattern", values="Recall").sort_index()
    data.to_csv(f"{OUT_ROOT}/versatility_all_measures.csv")
    data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
    heatmap(data, path, "Versatility per change pattern (sorted by mean recall)", "Change pattern",
            label="Recall", with_mean=True)


def plot_noise_curves(by_noise, path):
    d = by_noise.copy()
    d["Noise"] = pd.to_numeric(d["Noise Level"])
    sources = sorted(d["Log Source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources) + 2, 4.5), sharey=True, squeeze=False)
    for ax, source in zip(axes[0], sources):
        curves = d[d["Log Source"] == source].pivot(index="Noise", columns="Measure", values=HARMONIC_COLUMN).sort_index()
        order = curves.mean().sort_values(ascending=False).index
        for measure in order[TOP_N:]:
            ax.plot(curves.index, curves[measure], color="lightgray", linewidth=0.8)
        for measure in order[:TOP_N]:
            ax.plot(curves.index, curves[measure], marker="o", linewidth=1.8, label=measure)
        ax.set_title(source)
        ax.set_xlabel("Noise level")
        ax.set_xticks(curves.index)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Harmonic mean of F1, latency and versatility")
    axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), title=f"Top {TOP_N} (mean over noise)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noisy_vs_noiseless(versatility_by_noise, source, path):
    sub = versatility_by_noise[versatility_by_noise["Log Source"] == source]
    noiseless = sub[sub["Noise Level"] == "0"].pivot(index="Measure", columns="Change Pattern", values="Recall")
    noisy_rows = sub[sub["Noise Level"] == "noisy"]
    noisy = noisy_rows.pivot(index="Measure", columns="Change Pattern", values="Recall")
    if noiseless.empty or noisy.empty:
        return
    diff = noisy - noiseless
    diff = diff.loc[diff.mean(axis=1).sort_values(ascending=False).index]
    logs = noisy_rows.groupby("Change Pattern")["Logs"].max()
    diff.columns = [f"{cp} (n={logs[cp]})" for cp in diff.columns]
    heatmap(diff, path, f"{source}: recall with noise minus recall without noise", "Change pattern",
            cmap_name="RdBu", vmin=-1, vmax=1, label="Recall difference", with_mean=True)


if __name__ == "__main__":
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

    summary = collect_table(OUT_ROOT, "evaluation_measures.csv", ["Measure"])
    versatility = collect_table(OUT_ROOT, "versatility_per_change_pattern.csv", ["Measure", "Change Pattern"])
    by_noise = collect_table(OUT_ROOT, "evaluation_measures_by_noise.csv", ["Measure", "Log Source", "Noise Level"])
    versatility_by_noise = collect_table(OUT_ROOT, "versatility_per_change_pattern_by_noise.csv",
                                         ["Measure", "Log Source", "Noise Level", "Change Pattern"])

    if not summary.empty:
        summary.to_csv(f"{OUT_ROOT}/evaluation_all_measures.csv", index=False)
        plot_scorecard(summary, f"{PLOT_DIR}/scorecard.png")
        print(f"Scorecard: {len(summary)} measures")
    if not versatility.empty:
        plot_versatility(versatility, f"{PLOT_DIR}/versatility_heatmap.png")
        print(f"Versatility heatmap: {versatility['Measure'].nunique()} measures")
    if not by_noise.empty:
        by_noise.to_csv(f"{OUT_ROOT}/evaluation_by_noise_all_measures.csv", index=False)
        plot_noise_curves(by_noise, f"{PLOT_DIR}/noise_curves.png")
        print(f"Noise curves: {by_noise['Measure'].nunique()} measures")
    if not versatility_by_noise.empty:
        versatility_by_noise.to_csv(f"{OUT_ROOT}/versatility_by_noise_all_measures.csv", index=False)
        for source in sorted(versatility_by_noise["Log Source"].unique()):
            plot_noisy_vs_noiseless(versatility_by_noise, source, f"{PLOT_DIR}/noisy_vs_noiseless_{source}.png")
        print("Noisy vs noiseless plots saved")
    print(f"Plots in {PLOT_DIR}")