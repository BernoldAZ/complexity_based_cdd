from enum import Enum

from log_diversity_measures.log_representation import LogRepresentation


class SpeciesLogRepresentation(LogRepresentation, Enum):
    TRACE_VARIANT_SPECIES = 0
    N_GRAM_SPECIES = 1