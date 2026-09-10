"""Data ingestion, gridding and labelled fault injection.

`loaders` adapts real archives (GHCNh, Meteostat, generic CSV) to the canonical
`Observation` schema; `real` grids them into a `StationNetwork` and labels the
genuine extreme-weather spans; `injector` plants the labelled faults that the
evaluation is graded on.
"""

from .injector import InjectionResult, inject_faults, summarise_injection
from .loaders import filter_trusted, load_csv, load_ghcnh, load_meteostat
from .network import StationNetwork, dewpoint_from_rh, haversine_km
from .real import (
    detect_genuine_extremes,
    load_real_network,
    network_from_observations,
)

__all__ = [
    "StationNetwork",
    "haversine_km",
    "dewpoint_from_rh",
    "network_from_observations",
    "detect_genuine_extremes",
    "load_real_network",
    "InjectionResult",
    "inject_faults",
    "summarise_injection",
    "load_csv",
    "load_ghcnh",
    "load_meteostat",
    "filter_trusted",
]
