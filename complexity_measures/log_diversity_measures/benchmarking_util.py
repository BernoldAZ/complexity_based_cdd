import random as random
from datetime import datetime
from typing import Callable, Iterable

import pandas as pd
import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace
from tqdm import tqdm

from log_diversity_measures.diversity_measures import DiversityMeasures
from log_diversity_measures.external_log_representation import ExternalLogRepresentation
from log_diversity_measures.file_management_util import save_experiment_results
from log_diversity_measures.log_representation import LogRepresentation
from log_diversity_measures.species_log_representation import SpeciesLogRepresentation

RANGE_N_TUPLES_SPECIES = (1, 3)
COMPUTATIONALLY_EXPENSIVE_METRICS = {"avg_affinity", "avg_edit_distance", "num_acyclic_paths"}
UNPARAMETRIZED_SPECIES_TO_INCLUDE = {SpeciesLogRepresentation.TRACE_VARIANT_SPECIES}


def create_diversity_measure_obj(
        log: EventLog,
        range_n_tuple: tuple[int, int] | None = None,
        unparametrized_species_to_register: set["SpeciesLogRepresentation"] | None = None,
        measures_to_deregister: set[str] | None = None,
        species_measures_to_deregister: Iterable[str] | None = None
) -> "DiversityMeasures":
    """Construct and configure a DiversityMeasures instance.

    Args:
        log: The EventLog to analyze.
        range_n_tuple: Optional; a (start, end) tuple specifying the range of n‑tuple sizes to register.
            If None, defaults to RANGE_N_TUPLES_SPECIES.
        unparametrized_species_to_register: Optional; set of species representations to register without parameters.
            If None, defaults to UNPARAMETRIZED_SPECIES_TO_INCLUDE.
        measures_to_deregister: Optional; set of measure names to deregister (e.g. computationally expensive ones).
            If None, defaults to COMPUTATIONALLY_EXPENSIVE_METRICS.
        species_measures_to_deregister: Optional; iterable of species‑specific measures to deregister.
            If None, no species measures are explicitly deregistered.

    Returns:
        A configured DiversityMeasures object with specified n‑tuple and species registrations, and deregistered measures.
    """

    if range_n_tuple is None:
        range_n_tuple = RANGE_N_TUPLES_SPECIES
    if unparametrized_species_to_register is None:
        unparametrized_species_to_register = UNPARAMETRIZED_SPECIES_TO_INCLUDE
    if measures_to_deregister is None:
        measures_to_deregister = COMPUTATIONALLY_EXPENSIVE_METRICS

    dm = DiversityMeasures(log)
    if species_measures_to_deregister is not None:
        for measure in species_measures_to_deregister:
            dm.deregister_species_measure(measure)
    for n in range(*range_n_tuple):
        dm.register_n_tupel_species(n)
    for species in unparametrized_species_to_register:
        dm.register_unparametrized_species(species)
    for measure in measures_to_deregister:
        dm.deregister_measure(measure)
    return dm


def _measure_log_representations_creation_time(
        dm: DiversityMeasures,
        unparametrized_log_representations_to_measure: set[LogRepresentation] | None = None,
        n_tupel_log_representations_to_measure: Iterable[int] | None = None
) -> pd.DataFrame:
    if unparametrized_log_representations_to_measure is None:
        unparametrized_log_representations_to_measure = {*ExternalLogRepresentation,
                                                         SpeciesLogRepresentation.TRACE_VARIANT_SPECIES} - {
                                                            ExternalLogRepresentation.PM4PY_LOG}
    timed = dict()
    for log_representation in unparametrized_log_representations_to_measure:
        if log_representation == ExternalLogRepresentation.PM4PY_LOG:
            continue
        t0 = datetime.now()
        dm.get_unparametrized_log_representation(log_representation)
        timed[log_representation.name] = (datetime.now() - t0).total_seconds()
    for n in (n_tupel_log_representations_to_measure if n_tupel_log_representations_to_measure else range(
            *RANGE_N_TUPLES_SPECIES)):
        t0 = datetime.now()
        dm.get_n_gram_species(n)
        timed[f"{SpeciesLogRepresentation.N_GRAM_SPECIES.name}-{n}"] = (datetime.now() - t0).total_seconds()
    return pd.DataFrame([timed])


def sample_from_model(
        model: PetriNet,
        im: Marking,
        fm: Marking,
        sample_size: int,
        max_trace_length: int = 1000
) -> EventLog:
    """
    Sample traces from a Petri net model using pm4py playout simulation.

    Args:
        model: Petri net model to simulate.
        im: Initial marking.
        fm: Final marking.
        sample_size: Number of traces to generate.

    Returns:
        EventLog containing the sampled traces.
    """
    parameters = {'noTraces': sample_size, 'maxTraceLength': max_trace_length}
    log = pm4py.algo.simulation.playout.petri_net.algorithm.apply(
        model, im, fm, parameters,
        variant=pm4py.algo.simulation.playout.petri_net.algorithm.Variants.BASIC_PLAYOUT)
    for i in range(len(log)):
        while len(log[i]) == 0:
            tr = pm4py.algo.simulation.playout.petri_net.algorithm.apply(
                model, im, fm,
                parameters,
                variant=pm4py.algo.simulation.playout.petri_net.algorithm.Variants.BASIC_PLAYOUT)[0]
            log[i] = tr
    return log


def measure_diversity(dm: DiversityMeasures, diversity_measures: set[str] | None = None):
    """
    Measure diversity using the registered measures in a DiversityMeasures object and record computation time.
    """
    timed = dict()
    results = dict()
    for name, func in dm.registered_measures.items():
        if diversity_measures is not None and name not in diversity_measures:
            continue
        t0 = datetime.now()
        results[name] = func()
        timed[name] = (datetime.now() - t0).total_seconds()

    results_df = pd.DataFrame([results])
    timed_df = pd.DataFrame([timed])
    return results_df, timed_df


def sample_from_log(log: EventLog, n: int) -> list[Trace]:
    """
    Randomly sample n traces from an EventLog.
    """

    if n > len(log):
        Exception(f"sample size n={n}>log size={len(log)}")
    return random.sample(log, n)


def run_experiment(
        sampling_fun: Callable[[int], EventLog],
        number_of_samples_list: list[int],
        number_of_runs: int = 20,
        dataset_name_for_saving: str | None = None,
        measures_to_deregister: set[str] | None = None,
        measures_to_run: set[str] | None = None,
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a diversity experiment by sampling logs, measuring representations and diversity metrics multiple times.

    Args:
        sampling_fun: Function that takes an integer n and returns an EventLog of size n.
        number_of_samples_list: List of sample sizes to experiment with.
        number_of_runs: Number of repetitions for each sample size.
        dataset_name_for_saving: Optional dataset name for saving results.
        measures_to_deregister: Optional set of measure names to deregister.
        measures_to_run: Optional set of measure names to compute.
        verbose: Flag to enable verbose output.

    Returns:
        A tuple of DataFrames: (diversity measures, computation times, representation creation times).
    """
    all_div_measures = []
    all_calc_times = []
    all_repr_times = []

    for n in tqdm(number_of_samples_list):

        repr_creation_times = []
        measured_calculation_times = []
        diversity_measures = []

        for _ in range(number_of_runs):
            sampled_log = sampling_fun(n)
            dm = create_diversity_measure_obj(sampled_log, measures_to_deregister=measures_to_deregister)
            repr_creation_times.append(_measure_log_representations_creation_time(dm))
            diversity_results, diversity_calc_time_results = measure_diversity(dm, diversity_measures=measures_to_run)
            diversity_measures.append(diversity_results)
            measured_calculation_times.append(diversity_calc_time_results)

        div_df = pd.concat(diversity_measures, ignore_index=True)
        calc_df = pd.concat(measured_calculation_times, ignore_index=True)
        repr_df = pd.concat(repr_creation_times, ignore_index=True)

        div_df["n"] = n
        calc_df["n"] = n
        repr_df["n"] = n

        all_div_measures.append(div_df)
        all_calc_times.append(calc_df)
        all_repr_times.append(repr_df)

    div_measures_df = pd.concat(all_div_measures, ignore_index=True)
    calc_times_df = pd.concat(all_calc_times, ignore_index=True)
    repr_times_df = pd.concat(all_repr_times, ignore_index=True)
    if dataset_name_for_saving is not None:
        save_experiment_results(div_measures_df, dataset_name_for_saving, "diversity")
        save_experiment_results(calc_times_df, dataset_name_for_saving, "computation_time")
        save_experiment_results(repr_times_df, dataset_name_for_saving, "representation_time")
    return div_measures_df, calc_times_df, repr_times_df


def calculate_expected_risk(
        div_measures0: pd.DataFrame,
        div_measures1: pd.DataFrame,
        sample_size: int,
        x_axis: str,
        div_measures: list[str] | None = None
) -> pd.Series:
    """
    Calculate expected risk by comparing diversity measures between two DataFrames for a given sample size.

    Args:
        div_measures0: DataFrame with first set of diversity measures.
        div_measures1: DataFrame with second set of diversity measures.
        sample_size: Sample size to filter rows by.
        x_axis: Column name representing sample size.
        div_measures: Optional list of measure names to include in the calculation.

    Returns:
        Series with expected risk values for each measure.
    """

    div_measures0 = div_measures0[div_measures0[x_axis] == sample_size]
    div_measures1 = div_measures1[div_measures1[x_axis] == sample_size]
    if div_measures is not None:
        div_measures0 = div_measures0[div_measures]
        div_measures1 = div_measures1.loc[div_measures]

    cols = [c for c in div_measures0.columns if c != x_axis]

    counts = {}

    for col in cols:
        arr1 = div_measures0[col].to_numpy()[:, None]
        arr2 = div_measures1[col].to_numpy()[None, :]

        comparison = arr1 > arr2

        counts[col] = comparison.sum() / div_measures1.shape[0] ** 2

    return pd.Series(counts)


def calculate_all_pairwise_risks(
        div_measures0: pd.DataFrame,
        div_measures1: pd.DataFrame,
        x_axis: str,
        x_axis_values: list[int],
        div_measures: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute expected risk for all specified sample sizes and return as a DataFrame.

    Args:
        div_measures0: DataFrame with first set of diversity measures.
        div_measures1: DataFrame with second set of diversity measures.
        x_axis: Column name representing sample size.
        x_axis_values: List of sample size values to evaluate.
        div_measures: Optional list of measure names to include in the calculation.

    Returns:
        DataFrame containing expected risks for each measure and sample size.
    """
    results = []

    for x in x_axis_values:
        risk_series = calculate_expected_risk(div_measures0, div_measures1, x, x_axis, div_measures=div_measures)
        risk_series[x_axis] = x
        results.append(risk_series)

    risk_df = pd.DataFrame(results)

    return risk_df


def append_dataframe_columns(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Append columns from df_new to df_old ensuring row counts match.

    Args:
        df_old: Original DataFrame.
        df_new: DataFrame whose columns are to be appended.

    Raises:
        ValueError: If the number of rows in df_old and df_new do not match.

    Returns:
        Combined DataFrame with columns from both df_old and df_new.
    """
    if len(df_old) != len(df_new):
        raise ValueError(
            f"Row count mismatch: existing={len(df_old)}, new={len(df_new)}"
        )

    df_new = df_new.reset_index(drop=True)

    df_combined = pd.concat([df_old.reset_index(drop=True), df_new], axis=1)

    return df_combined
