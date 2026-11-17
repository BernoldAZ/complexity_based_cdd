import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from log_diversity_measures.measure_sources import get_metric_name, \
    get_species_dependent_metric, is_standard_diversity_metric


def _summarize_df(df: DataFrame) -> DataFrame:
    stats = df.groupby("n").agg(["mean", "count", "std"])

    new_cols = {}

    for c in df.columns:
        if c == "n":
            continue
        new_cols[(c, "ci_upper")] = stats[(c, "mean")] + stats[(c, "std")]
        new_cols[(c, "ci_lower")] = stats[(c, "mean")] - stats[(c, "std")]

    new_cols_df = pd.DataFrame(new_cols, index=stats.index)
    stats = pd.concat([stats, new_cols_df], axis=1)

    return stats


def rename_metrics(df: DataFrame) -> DataFrame:
    """
    Rename metric columns using reference names from the literature.
    """
    metric_names_from_source = {n: get_metric_name(n) if is_standard_diversity_metric(n)
    else get_species_dependent_metric(n) for n in df.columns}
    return df.rename(columns=metric_names_from_source)


def _keep_common_columns(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not dfs:
        return []

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    return [df.loc[:, list(common_cols)].copy() for df in dfs]


def plot_comparison(
        df_list: list[DataFrame],
        labels: list[str],
        baseline_dict: dict[str, dict[str, float]] | None = None,
        save_dir: str | None = None,
        rename: bool = True,
        dataset_name: str | None = None,
) -> None:
    """
    Plot metric comparisons across multiple DataFrames with optional baselines and saving.
    """
    df_list = _keep_common_columns(df_list)
    if rename:
        df_list = [rename_metrics(df) for df in df_list]
    stats_all = [_summarize_df(df) for df in df_list]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for c in df_list[0].columns:
        if c == 'n':
            continue

        plt.figure(figsize=(8, 5))
        model_colors = {}
        x_max = float('-inf')

        for i, stats in enumerate(stats_all):
            if stats[c, "mean"].isna().values.all():
                continue

            x_max = max(x_max, (max(stats.index) // 100) * 100)

            line = plt.plot(stats[c, "mean"], label=labels[i])[0]
            color = line.get_color()
            model_colors[labels[i]] = color

            plt.fill_between(
                stats.index,
                stats[c, "ci_lower"],
                stats[c, "ci_upper"],
                alpha=0.2,
                color=color
            )

        plt.xlabel('X-axis label', fontsize=15)
        plt.ylabel('Y-axis label', fontsize=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(fontsize=15)
        plt.tight_layout()

        if baseline_dict is not None and c in baseline_dict:
            for model_label, baseline_value in baseline_dict[c].items():
                if model_label in model_colors:
                    plt.axhline(
                        y=baseline_value,
                        linestyle='--',
                        alpha=0.7,
                        color=model_colors[model_label],
                    )

        plt.xticks(
            np.linspace(0, x_max, 6, dtype=int)
        )

        plt.xlabel("Log Size")
        plt.ylabel(c)
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            filename = c
            if dataset_name is not None:
                filename = f"{filename}_{dataset_name}"
            filename = f"{filename.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close()
        else:
            plt.show()


def plot_risk(
        risk_df: DataFrame,
        x_column: str,
        y_column: str
) -> None:
    """
    Plot a risk curve using specified x and y columns.
    """

    n0_values = risk_df[x_column]
    risk_values = risk_df[y_column]

    plt.figure(figsize=(8, 5))
    plt.plot(n0_values, risk_values, marker='o', linestyle='-')
    plt.xlabel(x_column)
    plt.ylabel(f"Risk ({y_column})")
    plt.title(f"Expected risk vs ({y_column})")
    plt.grid(True)
    plt.show()
