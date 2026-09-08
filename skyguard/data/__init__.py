"""Data generation and ingestion.

`simulator` builds a physically-grounded clean network; `injector` adds labelled
faults on top of it; `loaders` adapt real archives (GHCNh, Meteostat, CSV) to the
same canonical schema.
"""

from .injector import InjectionResult, inject_faults, summarise_injection
from .loaders import filter_trusted, load_csv, load_ghcnh, load_meteostat
from .simulator import (
    SimulatedNetwork,
    build_network,
    dewpoint_from_rh,
    haversine_km,
    simulate_network,
)

__all__ = [
    "SimulatedNetwork",
    "simulate_network",
    "build_network",
    "haversine_km",
    "dewpoint_from_rh",
    "InjectionResult",
    "inject_faults",
    "summarise_injection",
    "load_csv",
    "load_ghcnh",
    "load_meteostat",
    "filter_trusted",
]
