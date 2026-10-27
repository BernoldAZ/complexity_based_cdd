# Complexity-based Concept Drift Detection

KernelCPD complexity-measure approach: it runs KernelCPD drift
detection on the 151 CDRIFT logs (Bose, Ceravolo, Ostovar) with user-chosen complexity
measures, penalty and number of windows, and evaluates the detections with the unified
evaluation framework (F1, Latency, Versatility, Scalability, Parameter Sensitivity, Robustness).

## Contents

```
run_approach.py          Endpoint: detection + evaluation (run() or CLI)
run_experiments.py       Edit the variables at the bottom, then run it
cdrift/evaluation.py     Scoring utilities from cdrift-evaluation
complexity_measures/     Windowing + complexity vectors (only used for new window counts)
datasets/cdrift/         Logs, ground truth (drift_info.csv) and precomputed complexity
                         vectors for 75 (Bose), 100 (Ceravolo), 75 (Ostovar) windows
runs/<run_name>/         Outputs: config.json, detections.csv, evaluation_measures.csv,
                         versatility_per_change_pattern.csv, radar_chart_versatility.png
```

## Usage

Build the image once (from this folder, in PowerShell):

```
docker build -t cbcdd-image .
```

Run an experiment (edit `RUN_NAME`, `MEASURES`, `PEN`, `N_WINDOWS` in `run_experiments.py` first):

```
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_experiments.py
```

Or use the CLI directly:

```
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_approach.py --list-measures
docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_approach.py --run-name run_2_measures --measures lempel_ziv_complexity variant_entropy
```

Jupyter Lab (http://localhost:8887):

```
docker run -it --rm -p 8887:8887 -v "${PWD}:/workspace" cbcdd-image
```

and in a notebook at the repo root:

```python
from run_approach import run
results = run("run_2_measures", measures=["lempel_ziv_complexity", "variant_entropy"])
```

If `N_WINDOWS` differs from the precomputed 75/100/75, the complexity vectors are computed
once (slow) and cached in `datasets/cdrift/<dataset>/nw<N>/`.

## Sources

This folder is assembled from three repositories:

- **Approach** — the KernelCPD complexity-measure approach ("Selin"):
  https://github.com/selin-ada/selin-kernelcpd-cdrift
  Source of `run_approach.py`, `run_experiments.py`, `complexity_measures/` and the
  precomputed complexity vectors in `datasets/cdrift/`. The full repo also holds the research
  notebooks (`selin_final_pipeline.ipynb`, `step0_diagnostic.ipynb`,
  `evaluation_unified_framework.ipynb`) and proof files that are left out here.
- **Complexity measures** — the `log_diversity_measures` package by Daniel Kovacic and
  Martin Kabierski, from their `sample_based_process_comparison` repository:
  https://git01lab.cs.univie.ac.at/kabierskim94/sample_based_process_comparison.git
  It provides the 53 process-complexity measures computed per window. It is not included
  here; the Docker image clones and installs it, and it is only needed to compute vectors
  for new window counts.
- **Evaluation** — the CDRIFT evaluation framework (Adams et al., 2023; code by Cameron Pitsch):
  https://github.com/cpitsch/cdrift-evaluation
  `cdrift/evaluation.py` is shipped here unmodified; the rest of the package (`cdrift.utils`)
  is cloned by the Docker image and put on `PYTHONPATH`. The evaluation rules in
  `run_approach.py` are ported from `evaluation_unified_framework.ipynb` (lag window 200).
