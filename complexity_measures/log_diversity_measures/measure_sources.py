PAPER_VIDGOF = (
    "Vidgof, Maxim, Bastian Wurm, and Jan Mendling. "
    "\"The impact of process complexity on process performance: A study using event log data.\" "
    "International Conference on Business Process Management. Cham."
)
PAPER_SPECIES_DIVERSITY = """Martin Kabierski, Markus Richter, Matthias Weidlich,
Quantifying and relating the completeness and diversity of process representations using species estimation,
Information Systems,
Volume 130,
2025,
102512,
ISSN 0306-4379,
https://doi.org/10.1016/j.is.2024.102512."""

PAPER_SCHALK = """Schalk, Patrizia, and Artem Polyvyanyy. \"Mind the Gap: A
Formal
Investigation of the Relationship Between Log and Model
Complexity--Extended Version.\" arXiv preprint arXiv:2505.23233S"""
key_paper_name = "paper"
key_metric_name = "metric_name"


def get_metric_name(fun_name: str, source=PAPER_SCHALK, fail_silent: bool = True) -> str:
    if fun_name in METRIC_SOURCES:
        for source_dict in METRIC_SOURCES[fun_name]:
            if source_dict[key_paper_name] == source:
                return source_dict[key_metric_name]
    if not fail_silent:
        raise Exception(f"Metric name {fun_name} not found")
    return fun_name


def is_standard_diversity_metric(fun_name: str) -> bool:
    return fun_name in METRIC_SOURCES


def is_species_diversity_standard_diversity_metric(fun_name: str) -> bool:
    for fun_name0 in METRIC_SOURCES:
        if fun_name0 in fun_name:
            return True
    return False


def _get_species_name(species_name: str) -> str:
    if "n_gram_species" in species_name:
        return species_name.replace("n_gram_species_n", "N Gram Species; N=").replace("n_gram", "N=")
    if "trace_variant_species" in species_name:
        return species_name.replace("trace_variant_species", "Trace-Variant Species")
    else:
        raise Exception(f"Species name {species_name} is not recognized")


def get_species_dependent_metric(fun_name: str, source=PAPER_SPECIES_DIVERSITY, fail_silent: bool = True) -> str:
    trace_species_name = "trace_variant_species"
    n_gram_species_name = "n_gram_species"
    if n_gram_species_name in fun_name:
        idx = fun_name.find(n_gram_species_name)
        full_species_name = f"N Gram Species N={fun_name[-1]}"
        fun_name = fun_name[:idx - 1]
    elif trace_species_name in fun_name:
        idx = fun_name.find(trace_species_name)
        full_species_name = "Trace-Variant Species"
        fun_name = fun_name[:idx - 1]
    else:
        if fail_silent:
            return fun_name
        else:
            raise Exception(f"Species name {fun_name} not found")

    if fun_name in METRIC_SOURCES:
        for source_dict in METRIC_SOURCES[fun_name]:
            if source_dict[key_paper_name] == source:
                fun_name = source_dict[key_metric_name]

    return f"{full_species_name}: {fun_name}"


METRIC_SOURCES = {
    "num_events": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Events"},
                   {key_paper_name: PAPER_SCHALK, key_metric_name: "Magnitude"}],
    "num_event_types": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Event Types"},
                        {key_paper_name: PAPER_SCHALK, key_metric_name: "Variety"}],
    "num_sequences": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Sequences"},
                      {key_paper_name: PAPER_SCHALK, key_metric_name: "Length"}],
    "trace_length_min": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Minimum Sequence Length"},
                         {key_paper_name: PAPER_SCHALK, key_metric_name: "Minimum Trace Length"}],
    "trace_length_average": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Average Sequence Length"},
                             {key_paper_name: PAPER_SCHALK, key_metric_name: "Average Trace Length"}],
    "trace_length_max": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Maximum Sequence Length"},
                         {key_paper_name: PAPER_SCHALK, key_metric_name: "Maximum Trace Length"}],
    "num_acyclic_paths": [
        {key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Acyclic Paths in Transition Matrix"},
        {key_paper_name: PAPER_SCHALK, key_metric_name: "Level of Detail"}],
    "num_ties_in_transition_matrix": [
        {key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Ties in Transition Matrix"},
        {key_paper_name: PAPER_SCHALK, key_metric_name: "Number of Ties"}],
    "lempel_ziv_complexity": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Lempel-Ziv Complexity"},
                              {key_paper_name: PAPER_SCHALK, key_metric_name: "Lempel-Ziv complexity"}],
    "num_unique_sequences": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Number of Unique Sequences"},
                             {key_paper_name: PAPER_SCHALK, key_metric_name: "Number of Distinct Traces"}],
    "pct_unique_sequences": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Percentage of Unique Sequences"},
                             {key_paper_name: PAPER_SCHALK, key_metric_name: "Percentage of Distinct Traces"}],
    "average_distinct_events_per_trace": [{key_paper_name: PAPER_SCHALK, key_metric_name: "Structure"}],
    "avg_affinity": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Average Affinity"},
                     {key_paper_name: PAPER_SCHALK, key_metric_name: "Average Affinity"}],
    "deviation_from_random": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Deviation from Random"},
                              {key_paper_name: PAPER_SCHALK, key_metric_name: "Deviation from Random"}],
    "avg_edit_distance": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Average Edit Distance"},
                          {key_paper_name: PAPER_SCHALK, key_metric_name: "Average Edit-Distance"}],
    "sequence_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Sequence Entropy"},
                         {key_paper_name: PAPER_SCHALK, key_metric_name: "Sequence Entropy"}],
    "variant_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Variant Entropy"},
                        {key_paper_name: PAPER_SCHALK, key_metric_name: "Variant Entropy"}],
    "normalized_sequence_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Normalized Sequence Entropy"},
                                    {key_paper_name: PAPER_SCHALK, key_metric_name: "Normalized Sequence Entropy"}],
    "normalized_variant_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Normalized Variant Entropy"},
                                   {key_paper_name: PAPER_SCHALK, key_metric_name: "Normalized Variant Entropy"}],
    "enriched_sequence_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Enriched Sequence Entropy"},
                                  {key_paper_name: PAPER_SCHALK, key_metric_name: "Enriched Sequence Entropy"}],
    "enriched_variant_entropy": [{key_paper_name: PAPER_VIDGOF, key_metric_name: "Enriched Variant Entropy"},
                                 {key_paper_name: PAPER_SCHALK, key_metric_name: "Enriched Variant Entropy"}],
    "enriched_normalized_sequence_entropy": [
        {key_paper_name: PAPER_VIDGOF, key_metric_name: "Enriched Normalized Sequence Entropy"},
        {key_paper_name: PAPER_SCHALK, key_metric_name: "Enriched Normalized Sequence Entropy"}],
    "enriched_normalized_variant_entropy": [
        {key_paper_name: PAPER_VIDGOF, key_metric_name: "Enriched Normalized Variant Entropy"},
        {key_paper_name: PAPER_SCHALK, key_metric_name: "Enriched Normalized Variant Entropy"}],
    "species_richness_abundance": [{key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Richness(Abundance)"}],
    "exp_shannon_entropy_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Exponential Shannon Entropy(Abundance)"}],
    "simpson_diversity_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Simpson Diversity (Abundance)"}],
    "estimate_asymptotic_species_richness_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Asymptotic Richness(Abundance)"}],
    "estimate_asymptotic_exp_shannon_entropy_abundance": [{key_paper_name: PAPER_SPECIES_DIVERSITY,
                                                           key_metric_name: "Asymptotic Exponential Shannon Entropy(Abundance)"}],
    "estimate_asymptotic_simpson_diversity_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Asymptotic Simpson Diversity (Abundance)"}],
    "coverage": [{key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Coverage"}],
    "coverage_alternative": []
}
species_based_metric = {
    "species_richness_abundance": [{key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Richness(Abundance)"}],
    "exp_shannon_entropy_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Exponential Shannon Entropy(Abundance)"}],
    "simpson_diversity_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Simpson Diversity (Abundance)"}],
    "estimate_asymptotic_species_richness_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Asymptotic Richness(Abundance)"}],
    "estimate_asymptotic_exp_shannon_entropy_abundance": [{key_paper_name: PAPER_SPECIES_DIVERSITY,
                                                           key_metric_name: "Asymptotic Exponential Shannon Entropy(Abundance)"}],
    "estimate_asymptotic_simpson_diversity_abundance": [
        {key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Asymptotic Simpson Diversity (Abundance)"}],
    "coverage": [{key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Coverage"}],
    "coverage_alternative": [{key_paper_name: PAPER_SPECIES_DIVERSITY, key_metric_name: "Coverage Alternative"}]
}
