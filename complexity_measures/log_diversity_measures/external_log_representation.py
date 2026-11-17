from enum import Enum

from log_diversity_measures.log_representation import LogRepresentation


class ExternalLogRepresentation(LogRepresentation, Enum):
    PM4PY_LOG = 1
    EXTERNAL_LOG_REPRESENTATION = 2
    EXTERNAL_ENRICHED_LOG_REPRESENTATION = 3
    EXTENDED_PREFIX_AUTOMATA = 4
    ENRICHED_EXTENDED_PREFIX_AUTOMATA = 5