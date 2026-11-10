# Shared constants for the experiment scripts (run_cluster_experiments.py, run_emd_experiments.py,
# run_measure_sweep.py).


# Complexity measures that can be used (the duplicates of the 53 computed measures are commented out).
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
    "num_unique_sequences",
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
    #"coverage (n=(2,))",  # same as coverage_n_gram_species_n2
    #"coverage (n=(3,))",  # same as coverage_n_gram_species_n3
    #"coverage_alternative (n=(2,))",  # same as coverage_alternative_n_gram_species_n2
    #"coverage_alternative (n=(3,))",  # same as coverage_alternative_n_gram_species_n3
    #"estimate_asymptotic_exp_shannon_entropy_abundance (n=(2,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n2
    #"estimate_asymptotic_exp_shannon_entropy_abundance (n=(3,))",  # same as estimate_asymptotic_exp_shannon_entropy_abundance_n_gram_species_n3
    #"estimate_asymptotic_simpson_diversity_abundance (n=(2,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n2
    #"estimate_asymptotic_simpson_diversity_abundance (n=(3,))",  # same as estimate_asymptotic_simpson_diversity_abundance_n_gram_species_n3
    #"estimate_asymptotic_species_richness_abundance (n=(2,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n2
    #"estimate_asymptotic_species_richness_abundance (n=(3,))",  # same as estimate_asymptotic_species_richness_abundance_n_gram_species_n3
    #"exp_shannon_entropy_abundance (n=(2,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n2
    #"exp_shannon_entropy_abundance (n=(3,))",  # same as exp_shannon_entropy_abundance_n_gram_species_n3
    #"simpson_diversity_abundance (n=(2,))",  # same as simpson_diversity_abundance_n_gram_species_n2
    #"simpson_diversity_abundance (n=(3,))",  # same as simpson_diversity_abundance_n_gram_species_n3
    #"species_richness_abundance (n=(2,))",  # same as species_richness_abundance_n_gram_species_n2
    #"species_richness_abundance (n=(3,))",  # same as species_richness_abundance_n_gram_species_n3
]
