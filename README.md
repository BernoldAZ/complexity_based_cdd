# Complexity-based Concept Drift Detection


The repository adds the complexity measures to four existing drift detectors (the
baselines) and compares each of them with its standard version.

## Contents

```
constants.py                       ALL_MEASURES: the complexity measures used everywhere
cdrift/evaluation.py               Scoring utilities from cdrift-evaluation
complexity_measures/               Windowing + complexity vectors (only used for new window counts),
                                   including the log_diversity_measures package
datasets/cdrift/                   Logs, ground truth (drift_info.csv) and precomputed complexity
                                   vectors for 75 (Bose), 100 (Ceravolo), 75 (Ostovar) windows
runs/<run_name>/                   KernelCPD outputs: config.json, detections.csv, evaluation_measures.csv,
                                   versatility_per_change_pattern.csv, radar_chart_versatility.png

baselines/                         Code of the baseline detectors (see Baselines)
run_<alg>_experiments.py           One baseline run (alg = bose, adwinj, emd, cluster)
run_measure_sweep.py               Every baseline in "complexity" mode, once per measure
run_<alg>_patched_experiments.py   Patched runs with measures selected per change pattern
run_all_patched_experiments.py     Patched runs with all measures combined
baselines_results/                 All baseline runs and comparison plots (see Results)

analyze_results.py, compare_radar_charts.py, compile_patch_table.py,
build_standard_vs_complexity_table.py, compare_f1_standard.py   Figures and tables
prepare_unified_evaluation.py      Collects detections into evaluation_results/ for
                                   evaluation_unified_framework.ipynb
```

## Baselines

- **Bose** (`baselines/Bose_J/bose.py`): statistical tests on the J-measure (J) or window-count (WC)
  features of activity pairs over sliding windows. The J and WC variants are run separately.
- **ADWIN/J** (`baselines/Adwin_J/martjushev.py`, Martjushev et al.): adaptive windows with a statistical
  test on the J-measure, and the change point is located by recursive bisection.
- **EMD** (`baselines/EMD/earthmover.py`): Earth Mover's Distance between the stochastic languages of two
  adjacent windows, followed by peak detection.
- **Version Clustering (VC)** (`baselines/version-clustering-cdd-main/`): windows are clustered into process
  versions (HDBSCAN on DFG similarity vectors) and a drift is marked wherever the version changes.

Each baseline can run in three modes:
- **standard**: the original approach.
- **complexity**: the baseline's feature is replaced by the complexity measures.
- **combined**: both are used, weighted by `alpha` (0 = standard, 1 = complexity only). Bose and ADWIN/J use
  `p = (1 - alpha) * p_baseline + alpha * p_complexity`. EMD and VC use the `weighted_concat` fusion
  (see `baselines/EMD/emd_complexity_approach.py`). The patched runs use alpha ∈ {0.5, 0.3, 0.1, 0.05}.

## Complexity measures

37 measures from `constants.ALL_MEASURES`, computed per window with `log_diversity_measures`. The 53
computed measures include 16 duplicates, which are not used.

- **Log level**: `average_distinct_events_per_trace`, `avg_affinity`, `deviation_from_random`,
  `enriched_normalized_sequence_entropy`, `enriched_normalized_variant_entropy`, `enriched_sequence_entropy`,
  `enriched_variant_entropy`, `lempel_ziv_complexity`, `normalized_sequence_entropy`,
  `normalized_variant_entropy`, `num_event_types`, `num_events`, `num_sequences`, `num_unique_sequences`,
  `num_ties_in_transition_matrix`, `pct_unique_sequences`, `sequence_entropy`, `trace_length_average`,
  `trace_length_max`, `trace_length_min`, `variant_entropy`
- **2-activity sequences (n = 2)** and **3-activity sequences (n = 3)**, each as `<measure>_n_gram_species_n2/n3`:
  `coverage`, `coverage_alternative`, `estimate_asymptotic_exp_shannon_entropy_abundance`,
  `estimate_asymptotic_simpson_diversity_abundance`, `estimate_asymptotic_species_richness_abundance`,
  `exp_shannon_entropy_abundance`, `simpson_diversity_abundance`, `species_richness_abundance`

## Results (`baselines_results/`)

| Folder | Content |
|---|---|
| `bose_standard`, `bose_wc_standard`, `adwin_standard`, `emd_standard`, `vc_standard` | Standard baselines (alpha 0) |
| `bose_complexity_j`, `bose_complexity_wc`, `adwin_complexity` | Complexity-only runs |
| `sweep_<alg>/<measure>/` | One run per single measure. At the top: `f1_per_change_pattern.csv`, `f1_heatmap.png`, `measures_above_f1_<t>.csv`, `evaluation_all_measures.csv` |
| `patched_<alg>/` | One run per change pattern and alpha (`<ALG>_combined_<PATTERN>_alpha_<a>`), plus `measure_comparison.csv` and `selected_measures.json`. `patched_vc` only has the selection files (as no measures were selected, dont have more results) |
| `patched_<alg>_all_measures/` | All measures combined, one run per alpha |
| `comparison/` | Per algorithm: radar grids, standard vs. best patched radar, F1-delta plots. Plus `summary.csv`, `patch_summary.csv`, `patch_detail.csv`, `standard_radar_charts.pdf` |

Every run folder contains `config.json`, `detections.csv`, `evaluation_measures.csv`,
`versatility_per_change_pattern.csv` and `radar_chart_*.png`. The single-measure KernelCPD runs are in
`runs/`, and the unified framework results (accuracy, latency, versatility, ...) are in
`evaluation_results/Unified_framework_results/`.

### CDrift dataset
**Characteristics of the CDrift dataset. All drifts are sudden.**

| | **Bose** | **Ceravolo** | **Ostovar** |
|---|:---:|:---:|:---:|
| ***Event logs*** | | | |
| Event logs | 1 | 50 | 39 |
| Traces per log | 6,000 | 1,000 | 2,999 |
| Total traces | 6,000 | 50,000 | 116,961 |
| Noise levels (%) | 0 | 0, 5, 10, 15, 20 | 0, 2, 5 |
| Change patterns | - | 10 | 13 |
| ***Drifts*** | | | |
| Drifts per log | 4 | 1 | 2 |
| Drift positions | every 1,200 traces | trace 499 | traces 999, 1,999 |
| Total drifts | 4 | 50 | 78 |
| ***Change patterns*** | | | |
| Pattern codes | not given | cb, cd, cf, cp, lp, pl, pm, re, rp, sw | cb, cf, cm, cre, fr, lp, pl, pm, pre, rp, sm, sre, sw |

## Pipeline

For each algorithm, in this order:

1. Standard run (alpha 0), e.g. `run_bose_experiments.py` with `MODE = "bose"`.
2. Sweep over all single measures with `run_measure_sweep.py`.
3. Patched runs with `run_<algorithm>_patched_experiments.py`. It reads the standard run and the sweep,
   writes `measure_comparison.csv` and `selected_measures.json`, and runs one experiment per group and alpha.
   Finished runs are skipped, so the script can be restarted.
4. `analyze_results.py`, `compare_radar_charts.py` and `compile_patch_table.py` for the figures and tables.
   The paths of the standard and patched folders are set at the top of each script.

For Bose, set `BOSE_MEASURE` to `"J"` or `"WC"` in `run_bose_experiments.py` before running, and keep
the sweep and patched folders of the two variants separate.

## Usage

Build the image once (from this folder, in PowerShell):

```
docker build -t cbcdd-image .
```

### One baseline run

Each baseline has its own script. Edit the settings at the top of the script, then run it. The results
are written to `baselines_results/<RUN_NAME>/`.

| Baseline | Script | Mode variable: values | Main settings |
|---|---|---|---|
| Bose | `run_bose_experiments.py` | `MODE`: `"bose"`, `"complexity"`, `"combined"` | `BOSE_MEASURE` (`"J"` / `"WC"`), `TEST`, `ALPHA`, `MEASURES` |
| ADWIN/J | `run_adwinj_experiments.py` | `MODE`: `"adwin"`, `"complexity"`, `"combined"` | `TEST`, `THRESHOLD`, `MIN_WINDOW`, `MAX_WINDOW`, `ALPHA`, `MEASURES` |
| EMD | `run_emd_experiments.py` | `FEATURE_MODE`: `"standard"`, `"complexity"`, `"combined"` | `RUN_NAME`, `WINDOW_SIZE`, `STEP_SIZE`, `FUSION`, `ALPHA`, `MEASURES` |
| Version Clustering | `run_cluster_experiments.py` | `FEATURE_MODE`: `"similarity"`, `"complexity"`, `"combined"` | `PEN`, `N_WINDOWS`, `MIN_CLUSTER_SIZE`, `FUSION`, `ALPHA`, `MEASURES` |

The first mode value is the standard baseline. For Bose, ADWIN/J and VC the run name is derived from the mode
(e.g. `bose_standard`, `adwin_complexity`); for EMD set `RUN_NAME` yourself.

```
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_bose_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_adwinj_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_emd_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_cluster_experiments.py
```

### Sweep and patched runs

Set `APPROACHES` in `run_measure_sweep.py` (`"bose"`, `"bose_wc"`, `"adwinj"`, `"emd"`, `"vc"`); the
baseline settings are taken from the scripts above. Then run the patched script of the same baseline
(see Pipeline for the order):

```
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_measure_sweep.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_bose_patched_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_adwinj_patched_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_emd_patched_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_cluster_patched_experiments.py
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_all_patched_experiments.py
```

`run_all_patched_experiments.py` runs every baseline with all measures combined, for each alpha in `ALPHAS`.

### Jupyter Lab

```
docker run -it --rm -p 8887:8887 -v "${PWD}:/workspace" cbcdd-image
```

Then open http://localhost:8887.

### Complexity vectors

The complexity vectors are precomputed in `datasets/cdrift/<dataset>/` and `datasets/cdrift/<dataset>/nw<N>/`.
For a window count that has no vectors yet they are computed once (slow) and cached in `nw<N>/`. EMD computes them on its own sliding windows and caches them
in `datasets/cdrift/<dataset>/emd_w<WINDOW_SIZE>_s<STEP_SIZE>/`.
