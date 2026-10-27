# Small script to run one experiment with run_approach.run().
# Edit the variables below, then run from the repo root:  python run_experiments.py
#
# With Docker (from the repo root, in PowerShell):
#   docker build -t cbcdd-image .                                                  # once
#   docker run --rm -v "${PWD}:/workspace" cbcdd-image python run_experiments.py
# The -v flag mounts the repo into the container, so the container uses your local datasets
# and code and writes the results to runs/<run_name>/ in your repo on disk.
#
# Logs used per step (same filters as evaluation_unified_framework.ipynb):
#
#   Step                                              Logs  Ceravolo  Ostovar  Bose
#   Detection (KernelCPD runs on all)                  151        75       75     1
#   Atomic logs (basis of the whole evaluation)         90        50       39     1
#   F1 / Precision / Recall, Latency (noiseless)        24        10       13     1
#   Versatility (noiseless, no Bose)                    23        10       13     0
#   Robustness (all noise levels, no Bose)              89        50       39     0
#   Scalability, ms per case / event                    90        50       39     1
#
#   Robustness noise levels: Ceravolo 0/5/10/15/20 (10 logs each), Ostovar 0/2/5 (13 logs each).

from run_approach import run

# All 53 complexity measures that can be used.
ALL_MEASURES = [
    # Log-level measures
    "average_distinct_events_per_trace",
    "avg_affinity",
    "deviation_from_random",
    "enriched_normalized_sequence_entropy",
    "enriched_normalized_variant_entropy",
    "enriched_sequence_entropy",
    "enriched_variant_entropy",
    "lempel_ziv_complexity",
    "normalized_sequence_entropy",
    "normalized_variant_entropy",
    "num_event_types",
    "num_events",
    "num_sequences",
    "num_ties_in_transition_matrix",
    "pct_unique_sequences",
    "sequence_entropy",
    "trace_length_average",
    "trace_length_max",
    "trace_length_min",
    "variant_entropy",
    # Measures on 2-activity sequences
    "coverage_n_gram_species_n2",
    "coverage_alternative_n_gram_species_n2",
    "estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n2",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2",
    "estimate_asymptotic_species_richness_abundance_n_gram_species_n2",
    "exp_shannon_entropy_abundance_n_gram_species_n2",
    "simpson_diversity_abundance_n_gram_species_n2",
    "species_richness_abundance_n_gram_species_n2",
    # Measures on 3-activity sequences
    "coverage_n_gram_species_n3",
    "coverage_alternative_n_gram_species_n3",
    "estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n3",
    "estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3",
    "estimate_asymptotic_species_richness_abundance_n_gram_species_n3",
    "exp_shannon_entropy_abundance_n_gram_species_n3",
    "simpson_diversity_abundance_n_gram_species_n3",
    "species_richness_abundance_n_gram_species_n3",
    # Duplicates: dropped by the duplicate check on the Bose log (as done by Selin), HOWEVER can differ on other logs
    # because their values there are identical to a measure above.
    "num_unique_sequences",  # same as num_sequences
    "coverage (n=(2,))",  # same as coverage_n_gram_species_n2
    "coverage (n=(3,))",  # same as coverage_n_gram_species_n3
    "coverage_alternative (n=(2,))",  # same as coverage_alternative_n_gram_species_n2
    "coverage_alternative (n=(3,))",  # same as coverage_alternative_n_gram_species_n3
    "estimate_asymptotic_exp_shannon_entropy_abundance (n=(2,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n2
    "estimate_asymptotic_exp_shannon_entropy_abundance (n=(3,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n3
    "estimate_asymptotic_simpson_diversity_abundance (n=(2,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2
    "estimate_asymptotic_simpson_diversity_abundance (n=(3,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3
    "estimate_asymptotic_species_richness_abundance (n=(2,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n2
    "estimate_asymptotic_species_richness_abundance (n=(3,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n3
    "exp_shannon_entropy_abundance (n=(2,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n2
    "exp_shannon_entropy_abundance (n=(3,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n3
    "simpson_diversity_abundance (n=(2,))",  # same as simpson_diversity_abundance_n_gram_species_n2
    "simpson_diversity_abundance (n=(3,))",  # same as simpson_diversity_abundance_n_gram_species_n3
    "species_richness_abundance (n=(2,))",  # same as species_richness_abundance_n_gram_species_n2
    "species_richness_abundance (n=(3,))",  # same as species_richness_abundance_n_gram_species_n3
]

# ─── Inputs of the run ────────────────────────────────────────────────────────
# Note: RUN_NAME is used as the output folder name, so it can only contain letters, digits, '_' and '-'. 

RUN_NAME = "run_num_event_types"
MEASURES = ["num_event_types"]  # or ALL_MEASURES
PEN = {"Bose": 2.0, "Ceravolo": 2.0, "Ostovar": 2.0}
N_WINDOWS = {"Bose": 75, "Ceravolo": 100, "Ostovar": 75}
LAG_WINDOW = 200
DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "runs"

if __name__ == "__main__":
    results = run(
        RUN_NAME,
        measures=MEASURES,
        pen=PEN,
        n_windows=N_WINDOWS,
        lag_window=LAG_WINDOW,
        datasets_path=DATASETS_PATH,
        out_root=OUT_ROOT,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
