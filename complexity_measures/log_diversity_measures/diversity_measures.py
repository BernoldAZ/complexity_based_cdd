import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pm4py

#TODO make this more elegant
module_path = Path(__file__).resolve().parent.parent.parent / "deps/process-complexity"
sys.path.insert(0, str(module_path))
import Complexity as Complexity
import enriched as enriched

from log_diversity_measures.external_log_representation import ExternalLogRepresentation
from log_diversity_measures.log_representation import LogRepresentation
from log_diversity_measures.species_log_representation import SpeciesLogRepresentation
from pm4py.objects.log.obj import EventLog
from pm4py.util.string_distance import levenshtein_distance
from special4pm.estimation.metrics import estimate_species_richness_chao, \
    estimate_exp_shannon_entropy_abundance, estimate_simpson_diversity_abundance, hill_number, coverage
from special4pm.species import retrieve_species_trace_variant

def create_default_complexity_computer(log):
    diversity_measures = DiversityMeasures(log)
    diversity_measures.deregister_measure("avg_edit_distance")
    diversity_measures.deregister_measure("num_acyclic_paths")
    return diversity_measures



def measure(func):
    func._is_measure = True
    return func


def species_dependent_measure(func):
    func._is_species_dependent_measure = True
    return func


def _retrieve_species_n_gram_adjusted(trace, n):
    events = [x['concept:name'] for x in trace]
    if len(events) < n:
        events.extend("<>" for _ in range(n - len(events)))
    return [",".join(events[x:x + n]) for x in range(0, len(events) - n + 1)]


class DiversityMeasures:

    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.registered_measures = {name: getattr(self, name) for name in dir(self)
                                    if getattr(getattr(self, name), "_is_measure", False)}
        self.species_dependent_measures = {name: getattr(self, name) for name in dir(self)
                                           if getattr(getattr(self, name), "_is_species_dependent_measure", False)}
        self.representations: dict[LogRepresentation, Any] = {}
        self.parametrized_representations: dict[LogRepresentation, dict[tuple[Any, ...], Any]] = {
            SpeciesLogRepresentation.N_GRAM_SPECIES: dict()}

    def __generate_external_log_representation(self):
        return Complexity.generate_log(self.event_log)

    def __generate_enriched_external_log_representation(self):
        return enriched.generate_enriched_log(self.event_log)

    def __generate_extended_prefix_automata_representation(self):
        return Complexity.build_graph(
            self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTERNAL_LOG_REPRESENTATION))

    def __generate_enriched_extended_prefix_automata_representation(self):
        enriched_external_log_representation = self.get_unparametrized_log_representation(
            ExternalLogRepresentation.EXTERNAL_ENRICHED_LOG_REPRESENTATION)
        return enriched.build_enriched_graph(enriched_external_log_representation)

    def __generate_n_gram_species_representation(self, n: int):
        return dict(
            Counter(n_gram for trace in self.event_log for n_gram in _retrieve_species_n_gram_adjusted(trace, n)))

    def __generate_trace_species(self):
        return dict(Counter(retrieve_species_trace_variant(trace)[0] for trace in self.event_log))

    def __get_trace_strs(self) -> list[str]:
        activity_names = list(set(e["concept:name"] for t in self.event_log for e in t))
        activity_name_alph_map = {n: chr(i) for i, n in enumerate(activity_names)}
        return ["".join([activity_name_alph_map[e["concept:name"]] for e in t]) for t in self.event_log]

    def __generate_unparametrized_representations(self, log_representation: LogRepresentation):
        representation_generator: dict[LogRepresentation, Callable] = {
            ExternalLogRepresentation.EXTERNAL_LOG_REPRESENTATION: self.__generate_external_log_representation,
            ExternalLogRepresentation.EXTERNAL_ENRICHED_LOG_REPRESENTATION: self.__generate_enriched_external_log_representation,
            ExternalLogRepresentation.EXTENDED_PREFIX_AUTOMATA: self.__generate_extended_prefix_automata_representation,
            ExternalLogRepresentation.ENRICHED_EXTENDED_PREFIX_AUTOMATA: self.__generate_enriched_extended_prefix_automata_representation,
            SpeciesLogRepresentation.TRACE_VARIANT_SPECIES: self.__generate_trace_species
        }
        self.representations[log_representation] = representation_generator[log_representation]()

    def get_unparametrized_log_representation(self, log_representation: LogRepresentation):
        if log_representation not in self.representations:
            self.__generate_unparametrized_representations(log_representation)
        return self.representations[log_representation]

    def get_n_gram_species(self, n: int):
        if not (n,) in self.parametrized_representations[SpeciesLogRepresentation.N_GRAM_SPECIES]:
            self.parametrized_representations[SpeciesLogRepresentation.N_GRAM_SPECIES][
                (n,)] = self.__generate_n_gram_species_representation(n)
        return self.parametrized_representations[SpeciesLogRepresentation.N_GRAM_SPECIES][(n,)]

    @measure
    def num_events(self, verbose: bool = False) -> float:
        return Complexity.measure_magnitude(
            self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTERNAL_LOG_REPRESENTATION),
            quiet=not verbose, verbose=verbose)

    @measure
    def num_event_types(self, verbose: bool = False) -> float:
        return Complexity.measure_variety(self.event_log, quiet=not verbose, verbose=verbose)

    @measure
    def num_sequences(self, verbose: bool = False) -> float:
        return Complexity.measure_support(self.event_log, quiet=not verbose, verbose=verbose)

    @measure
    def trace_length_min(self, verbose: bool = False) -> float:
        result = min(len(t) for t in self.event_log)
        if verbose:
            print(f"Average Trace Length: {result}")
        return result

    @measure
    def trace_length_average(self, verbose: bool = False) -> float:
        result = sum(len(t) for t in self.event_log) / len(self.event_log)
        if verbose:
            print(f"Average Trace Length: {result}")
        return result

    @measure
    def trace_length_max(self, verbose: bool = False) -> float:
        result = max(len(t) for t in self.event_log)
        if verbose:
            print(f"Max Trace Length: {result}")
        return result

    @measure
    def num_acyclic_paths(self, verbose: bool = False) -> float:
        dfg, start_activities, end_activities = pm4py.discover_dfg(self.event_log)
        G = nx.DiGraph()
        for (src, tgt), weight in dfg.items():
            G.add_edge(src, tgt, weight=weight)

        start_nodes = list(start_activities.keys())
        end_nodes = list(end_activities.keys())

        number_of_paths = 0
        for s in start_nodes:
            for e in end_nodes:
                number_of_paths += len(list(nx.all_simple_paths(G, s, e)))
        if verbose:
            print("Number of simple paths:", number_of_paths)
        return number_of_paths

    @measure
    def num_ties_in_transition_matrix(self, verbose: bool = False) -> float:
        df_pairs = set()
        for trace in self.event_log:
            trace = list(map(lambda e: e["concept:name"], sorted(trace, key=lambda e: e["time:timestamp"])))
            df_pairs.update(zip(trace, trace[1:]))
        result = len(df_pairs.difference(map(lambda pair: (pair[1], pair[0]), df_pairs)))
        if verbose:
            print(f"Number of Ties in Transition Matrix: {result}")
        return result

    @measure
    def lempel_ziv_complexity(self, verbose: bool = False) -> float:
        return Complexity.measure_lempel_ziv(
            self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTERNAL_LOG_REPRESENTATION),
            quiet=not verbose, verbose=verbose)

    @measure
    def num_unique_sequences(self, verbose: bool = False) -> float:
        result = len(set(tuple([e['concept:name'] for e in t]) for t in self.event_log))
        if verbose:
            print(f"Number of unique sequences: {result}")
        return result

    @measure
    def pct_unique_sequences(self, verbose: bool = False) -> float:
        result = self.num_unique_sequences() * 100 / len(self.event_log)
        if verbose:
            print("Percentage of Unique Sequences: ", result)
        return result

    @measure
    def average_distinct_events_per_trace(self, verbose: bool = False) -> float:
        result = sum(len(set(map(lambda e: e["concept:name"], trace))) for trace in self.event_log) / sum(
            1 for _ in self.event_log)
        if verbose:
            print(f"Average distinct event ratio per trace: {result}")
        return result

    @measure
    def avg_affinity(self, verbose: bool = False) -> float:
        def get_affinity_set(trace_str: str):
            return {trace_str[i] + trace_str[i + 1] for i in range(len(trace_str) - 1)}

        trace_strs = self.__get_trace_strs()
        aff_ctrs = [get_affinity_set(t) for t in trace_strs]
        aff_sum = 0
        for i in range(len(trace_strs)):
            for j in range(i):
                intersec_size = len(aff_ctrs[i].intersection(aff_ctrs[j]))
                aff_sum += (intersec_size / len(aff_ctrs[i].union(aff_ctrs[j]))) if intersec_size > 0 else 0
        normalized_affinity = aff_sum / (len(trace_strs) * (len(trace_strs) - 1) / 2) if len(trace_strs) > 1 else 0
        if verbose:
            print(f"Average Affinity: {normalized_affinity}")
        return normalized_affinity

    @measure
    def deviation_from_random(self, verbose: bool = False) -> float:
        return Complexity.measure_deviation_from_random(
            self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTERNAL_LOG_REPRESENTATION),
            self.event_log,
            quiet=not verbose, verbose=verbose)

    @measure
    def avg_edit_distance(self, verbose: bool = False) -> float:
        trace_strs = self.__get_trace_strs()
        total_edit_distance = 0
        for i in range(len(trace_strs)):
            for j in range(i):
                total_edit_distance += levenshtein_distance(trace_strs[i], trace_strs[j])
        avg_edit_distance = total_edit_distance / (len(trace_strs) * (len(trace_strs) - 1) / 2) if len(
            trace_strs) > 1 else 0

        if verbose:
            print(f"Average Edit Distance: {avg_edit_distance}")

        return avg_edit_distance

    @measure
    def sequence_entropy(self, verbose: bool = False) -> float:
        epa = self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTENDED_PREFIX_AUTOMATA)
        return Complexity.log_complexity(epa)[0]

    @measure
    def variant_entropy(self, verbose: bool = False) -> float:
        epa = self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTENDED_PREFIX_AUTOMATA)
        return Complexity.graph_complexity(epa)[0]

    @measure
    def normalized_sequence_entropy(self, verbose: bool = False) -> float:
        epa = self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTENDED_PREFIX_AUTOMATA)
        return Complexity.log_complexity(epa)[1]

    @measure
    def normalized_variant_entropy(self, verbose: bool = False) -> float:
        epa = self.get_unparametrized_log_representation(ExternalLogRepresentation.EXTENDED_PREFIX_AUTOMATA)
        return Complexity.graph_complexity(epa)[1]

    @measure
    def enriched_sequence_entropy(self, verbose: bool = False) -> float:
        eepa = self.get_unparametrized_log_representation(ExternalLogRepresentation.ENRICHED_EXTENDED_PREFIX_AUTOMATA)
        return Complexity.log_complexity(eepa)[0]

    @measure
    def enriched_variant_entropy(self, verbose: bool = False) -> float:
        eepa = self.get_unparametrized_log_representation(ExternalLogRepresentation.ENRICHED_EXTENDED_PREFIX_AUTOMATA)
        return Complexity.graph_complexity(eepa)[0]

    @measure
    def enriched_normalized_sequence_entropy(self, verbose: bool = False) -> float:
        eepa = self.get_unparametrized_log_representation(ExternalLogRepresentation.ENRICHED_EXTENDED_PREFIX_AUTOMATA)
        return Complexity.log_complexity(eepa)[1]

    @measure
    def enriched_normalized_variant_entropy(self, verbose: bool = False) -> float:
        eepa = self.get_unparametrized_log_representation(ExternalLogRepresentation.ENRICHED_EXTENDED_PREFIX_AUTOMATA)
        return Complexity.graph_complexity(eepa)[1]

    @species_dependent_measure
    def species_richness_abundance(self, species_representation: dict[str, int]):
        return hill_number(0, species_representation)

    @species_dependent_measure
    def exp_shannon_entropy_abundance(self, species_representation: dict[str, int]):
        return hill_number(1, species_representation)

    @species_dependent_measure
    def simpson_diversity_abundance(self, species_representation: dict[str, int]):
        return hill_number(2, species_representation)

    @species_dependent_measure
    def estimate_asymptotic_species_richness_abundance(self, species_representation: dict[str, int]):
        return estimate_species_richness_chao(species_representation)

    @species_dependent_measure
    def estimate_asymptotic_exp_shannon_entropy_abundance(self, species_representation: dict[str, int]):
        return estimate_exp_shannon_entropy_abundance(
            self.get_unparametrized_log_representation(SpeciesLogRepresentation.TRACE_VARIANT_SPECIES),
            sum(c for c in species_representation.values()))

    @species_dependent_measure
    def estimate_asymptotic_simpson_diversity_abundance(self, species_representation: dict[str, int]):
        return estimate_simpson_diversity_abundance(species_representation,
                                                    sum(c for c in species_representation.values()))

    @species_dependent_measure
    def coverage(self, species_representation: dict[str, int]):
        return coverage(species_representation, sum(c for c in species_representation.values()))

    @species_dependent_measure
    def coverage_alternative(self, species_representation: dict[str, int]):
        return 1 - len(species_representation.keys()) / sum(species_representation.values())

    def register_measure(self, measure_name: str):
        if hasattr(self, measure_name):
            self.registered_measures[measure_name] = getattr(self, measure_name)
        else:
            raise Exception(f'No method with name "{measure_name}" in class.')

    def deregister_measure(self, measure_name: str):
        if measure_name in self.registered_measures:
            del self.registered_measures[measure_name]
        else:
            raise Exception(f'No registered measure with name "{measure_name}".')

    def deregister_species_measure(self, measure_name: str):
        if measure_name in self.species_dependent_measures:
            del self.species_dependent_measures[measure_name]
        else:
            raise Exception(f'No species dependent measure method with name "{measure_name}" in class.')

    def register_unparametrized_species(self, species_log_representation: SpeciesLogRepresentation):
        for method_id, method in self.species_dependent_measures.items():
            self.registered_measures[f"{method_id}_{species_log_representation.name.lower()}"] = lambda m=method: m(
                self.get_unparametrized_log_representation(species_log_representation))

    def register_n_tupel_species(self, n):
        for method_id, method in self.species_dependent_measures.items():
            self.registered_measures[
                f"{method_id}_{SpeciesLogRepresentation.N_GRAM_SPECIES.name.lower()}_n{n}"] = lambda m=method: m(
                self.get_n_gram_species(n))
