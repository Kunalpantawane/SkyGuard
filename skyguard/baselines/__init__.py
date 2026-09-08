"""Classical baselines for the ablation ladder.

Each baseline fits on clean levels and scores points in [0, 1] through the
same empirical-CDF calibration below, so the harness compares detectors, not
their private score scales. These exist to be beaten honestly: if the hybrid
cannot outscore z-scores on spikes, something is wrong with the hybrid.
"""

from __future__ import annotations

import numpy as np


def fit_ecdf(clean_scores: np.ndarray) -> np.ndarray:
    """Sorted clean scores: the calibration reference for one detector."""
    arr = np.sort(np.asarray(clean_scores, dtype=np.float64).ravel())
    if arr.size == 0:
        raise ValueError("need clean scores to calibrate a baseline")
    return arr


def ecdf_scores(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of clean scores at or below each value — always in [0, 1]."""
    vals = np.asarray(values, dtype=np.float64)
    ranks = np.searchsorted(calibration, vals, side="right") / max(len(calibration), 1)
    return np.clip(ranks, 0.0, 1.0)
