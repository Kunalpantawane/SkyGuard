"""Isolation Forest baseline, numpy-only.

Random splits isolate outliers in few steps: the anomaly score is the classic
2^(-E[depth]/c(n)). Subsampled trees keep it fast; ECDF on clean data keeps
it comparable. Catches isolated outliers, misses structured faults like drift
— the gap the temporal model is meant to close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import ecdf_scores, fit_ecdf


def _c_factor(n: int) -> float:
    """Average unsuccessful-search path length in a BST (the IF normaliser)."""
    if n <= 1:
        return 1.0
    return 2.0 * (float(np.log(n - 1)) + float(np.euler_gamma)) - 2.0 * (n - 1) / n


@dataclass
class _INode:
    feature: int = -1
    split: float = 0.0
    left: object = None
    right: object = None
    size: int = 0


class IsolationForest:
    """Bagged isolation trees over the three raw variables."""

    def __init__(self, n_trees: int = 50, max_depth: int = 8,
                 subsample: int = 256, seed: int = 0) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample = subsample
        self.seed = seed
        self.trees: List[_INode] = []
        self.calibration: np.ndarray = np.array([0.0])
        self._c_n: float = 1.0

    def fit(self, clean_levels: np.ndarray) -> "IsolationForest":
        """Grow trees on clean (N, 3) levels and calibrate on them."""
        mat = np.asarray(clean_levels, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != 3 or mat.shape[0] == 0:
            raise ValueError("clean levels must be non-empty (N, 3)")
        rng = np.random.default_rng(self.seed)
        self.trees = [
            self._grow(mat[rng.choice(mat.shape[0], size=min(self.subsample, mat.shape[0]),
                                      replace=False)], rng, 0)
            for _ in range(self.n_trees)
        ]
        self._c_n = _c_factor(min(self.subsample, mat.shape[0]))
        self.calibration = fit_ecdf(self.raw_scores(mat))
        return self

    def _grow(self, sample: np.ndarray, rng: np.random.Generator, depth: int) -> _INode:
        if depth >= self.max_depth or sample.shape[0] <= 1:
            return _INode(size=sample.shape[0])
        feat = int(rng.integers(0, sample.shape[1]))
        lo, hi = float(sample[:, feat].min()), float(sample[:, feat].max())
        if lo == hi:
            return _INode(size=sample.shape[0])
        split = float(rng.uniform(lo, hi))
        mask = sample[:, feat] < split
        if mask.sum() == 0 or mask.sum() == sample.shape[0]:
            return _INode(size=sample.shape[0])
        node = _INode(feature=feat, split=split, size=sample.shape[0])
        node.left = self._grow(sample[mask], rng, depth + 1)
        node.right = self._grow(sample[~mask], rng, depth + 1)
        return node

    def _depth(self, point: np.ndarray, node: _INode, depth: int) -> float:
        while node.left is not None:
            assert isinstance(node.left, _INode) and isinstance(node.right, _INode)
            if point[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
            depth += 1
        return depth + _c_factor(node.size)

    def raw_scores(self, levels: np.ndarray) -> np.ndarray:
        """Mean isolation score per row, uncalibrated (higher = stranger)."""
        mat = np.asarray(levels, dtype=np.float64)
        depths = np.array([[self._depth(row, tree, 0) for tree in self.trees] for row in mat])
        return np.power(2.0, -depths.mean(axis=1) / self._c_n)

    def score_points(self, levels: np.ndarray) -> np.ndarray:
        """ECDF-calibrated [0, 1] anomaly scores."""
        if not self.trees:
            raise ValueError("forest is not fitted")
        return ecdf_scores(self.calibration, self.raw_scores(levels))
