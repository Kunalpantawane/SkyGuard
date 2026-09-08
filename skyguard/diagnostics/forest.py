"""Layer 6a — numpy random forest fault classifier.

Why a forest instead of a neural classifier: it consumes interpretable
engineered features (the diagnostic fingerprint of each fault type) and pairs
naturally with Shapley — each tree is a set of readable rules, and the forest
average is a calibrated vote. Gini splits, bootstrap bagging and per-split
feature subsampling, all in numpy so the edge and CPU story stays intact.

Trained on injector labels (`context/data.md` taxonomy); evaluated by the
fault-class confusion matrix (`context/evaluation.md`), never by accuracy
alone — rare classes must not be drowned out, hence the balancing option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import DEFAULT_CONFIG, ForestConfig
from ..types import FaultClass

# Fixed diagnostic fingerprint, in column order. Trees split on positions, so
# this order is part of the model contract — never reorder, only append.
FEATURE_NAMES: Tuple[str, ...] = (
    "recon_temp",
    "recon_pressure",
    "recon_rh",
    "rate_of_change",
    "repeat_run",
    "variance_ratio",
    "drift_slope",
    "mahalanobis",
    "spatial_residual",
    "missing_run",
    "sentinel_flag",
    "recent_anomaly_rate",
    "hour_sin",
    "hour_cos",
)

N_FEATURES = len(FEATURE_NAMES)

# Fault classes the classifier can emit. Mirrors the injector taxonomy;
# GENUINE_EXTREME is deliberately included — "real weather, trust it" is a
# diagnosis the system must be able to reach, not just a lack of alarm.
FAULT_ORDER: Tuple[FaultClass, ...] = (
    FaultClass.NONE,
    FaultClass.SPIKE,
    FaultClass.DROP,
    FaultClass.STUCK,
    FaultClass.DRIFT,
    FaultClass.STEP,
    FaultClass.NOISE,
    FaultClass.MISSING,
    FaultClass.CORRUPT,
    FaultClass.MULTIVARIATE,
    FaultClass.SPATIAL,
    FaultClass.GENUINE_EXTREME,
)

_LEAF = -2
_MAX_CANDIDATE_THRESHOLDS = 64


@dataclass
class _Tree:
    """One CART tree as parallel arrays (vectorisable, serialisable)."""

    feature: np.ndarray      # (nodes,) split feature, _LEAF for leaves
    threshold: np.ndarray    # (nodes,) split point
    left: np.ndarray         # (nodes,) left child index
    right: np.ndarray        # (nodes,) right child index
    proba: np.ndarray        # (nodes, K) class distribution at node
    n_nodes: int = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Route every row to a leaf; returns (N, K) distributions."""
        out = np.empty((X.shape[0], self.proba.shape[1]), dtype=np.float64)
        for r in range(X.shape[0]):
            node = 0
            while self.feature[node] != _LEAF:
                node = self.left[node] if X[r, self.feature[node]] <= self.threshold[node] else self.right[node]
            out[r] = self.proba[node]
        return out


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _grow(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    n_classes: int, rng: np.random.Generator, config: ForestConfig, depth: int,
    feature: List[int], threshold: List[float], left: List[int], right: List[int],
    proba: List[np.ndarray],
) -> int:
    """Grow one subtree; returns its root node index."""
    n = X.shape[0]
    counts = np.bincount(y, weights=weights, minlength=n_classes)
    node = len(proba)
    proba.append(counts / max(counts.sum(), 1e-12))
    feature.append(_LEAF)
    threshold.append(0.0)
    left.append(-1)
    right.append(-1)

    if (
        depth >= config.max_depth
        or n < config.min_samples_split
        or _gini(counts) <= 0.0
    ):
        return node

    n_try = _mtry(X.shape[1], config.max_features, rng)
    candidates = rng.choice(X.shape[1], size=n_try, replace=False)
    best_gain, best_feat, best_thr = 0.0, -1, 0.0
    parent_gini = _gini(counts)
    for feat in candidates:
        col = X[:, int(feat)]
        order = np.argsort(col, kind="stable")
        values = col[order]
        labels = y[order]
        w = weights[order]
        distinct = np.flatnonzero(np.diff(values))
        if distinct.size == 0:
            continue
        # Cap the split search: evenly spaced candidate cut points keep the
        # fit O(N log N) without changing which splits win on real data.
        if distinct.size > _MAX_CANDIDATE_THRESHOLDS:
            distinct = distinct[np.linspace(0, distinct.size - 1, _MAX_CANDIDATE_THRESHOLDS).astype(int)]
        for cut in distinct:
            left_counts = np.bincount(labels[: int(cut) + 1], weights=w[: int(cut) + 1], minlength=n_classes)
            right_counts = counts - left_counts
            if left_counts.sum() < config.min_samples_leaf or right_counts.sum() < config.min_samples_leaf:
                continue
            gain = parent_gini - (
                left_counts.sum() / n * _gini(left_counts)
                + right_counts.sum() / n * _gini(right_counts)
            )
            if gain > best_gain:
                best_gain = gain
                best_feat = int(feat)
                best_thr = float((values[cut] + values[cut + 1]) / 2.0)

    if best_feat < 0:
        return node
    go_left = X[:, best_feat] <= best_thr
    if go_left.sum() == 0 or go_left.sum() == n:
        return node
    feature[node] = best_feat
    threshold[node] = best_thr
    left[node] = _grow(X[go_left], y[go_left], weights[go_left], n_classes, rng, config, depth + 1,
                       feature, threshold, left, right, proba)
    right[node] = _grow(X[~go_left], y[~go_left], weights[~go_left], n_classes, rng, config, depth + 1,
                        feature, threshold, left, right, proba)
    return node


def _mtry(n_features: int, max_features: str, rng: np.random.Generator) -> int:
    """Per-split feature budget. Subsampling decorrelates the trees, which is
    where the forest's robustness (and Shapley's signal) comes from."""
    if max_features == "sqrt":
        return max(1, int(round(float(np.sqrt(n_features)))))
    return n_features


class RandomForestClassifier:
    """Bagged CART forest over integer labels. Generic engine; the fault
    taxonomy wrapper below fixes the class order and feature contract."""

    def __init__(self, n_classes: int, config: ForestConfig | None = None, seed: int | None = None) -> None:
        self.config: ForestConfig = config or DEFAULT_CONFIG.forest
        self.n_classes = int(n_classes)
        self.seed = self.config.seed if seed is None else seed
        self.trees: List[_Tree] = []
        self.n_features_in: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit on integer labels in [0, n_classes). Deterministic given seed."""
        mat = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y, dtype=int).ravel()
        if mat.shape[0] != labels.shape[0] or mat.shape[0] == 0:
            raise ValueError("X and y must share a non-zero row count")
        if labels.min() < 0 or labels.max() >= self.n_classes:
            raise ValueError(f"labels must lie in [0, {self.n_classes})")
        if not np.all(np.isfinite(mat)):
            raise ValueError("forest features must be finite")
        self.n_features_in = mat.shape[1]
        rng = np.random.default_rng(self.seed)
        base_weights = np.ones(mat.shape[0], dtype=np.float64)
        if self.config.class_balance:
            freq = np.bincount(labels, minlength=self.n_classes).astype(np.float64)
            base_weights = len(labels) / (self.n_classes * np.maximum(freq[labels], 1.0))
        self.trees = []
        for _ in range(self.config.n_trees):
            draw = rng.integers(0, mat.shape[0], size=mat.shape[0])
            feature: List[int] = []
            threshold: List[float] = []
            left: List[int] = []
            right: List[int] = []
            proba: List[np.ndarray] = []
            _grow(mat[draw], labels[draw], base_weights[draw], self.n_classes,
                  rng, self.config, 0, feature, threshold, left, right, proba)
            self.trees.append(_Tree(
                feature=np.array(feature, dtype=int),
                threshold=np.array(threshold, dtype=np.float64),
                left=np.array(left, dtype=int),
                right=np.array(right, dtype=int),
                proba=np.array(proba, dtype=np.float64),
                n_nodes=len(proba),
            ))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mean leaf distribution across trees; rows sum to 1."""
        if not self.trees:
            raise ValueError("forest is not fitted")
        mat = np.asarray(X, dtype=np.float64)
        if mat.shape[1] != self.n_features_in:
            raise ValueError(
                f"expected {self.n_features_in} features, got {mat.shape[1]}"
            )
        votes = np.mean([tree.predict_proba(mat) for tree in self.trees], axis=0)
        return votes / np.maximum(votes.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority-vote labels."""
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class FaultPrediction:
    """One diagnosis: the class, its confidence, and the full distribution."""

    fault_class: FaultClass
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


class FaultClassifier:
    """Layer 6 entry point: fingerprint vector -> fault class + confidence."""

    def __init__(self, config: ForestConfig | None = None, seed: int | None = None) -> None:
        self.config: ForestConfig = config or DEFAULT_CONFIG.forest
        self.forest = RandomForestClassifier(n_classes=len(FAULT_ORDER), config=self.config, seed=seed)
        self.is_fitted = False

    def fit(self, X: np.ndarray, labels: Sequence[FaultClass]) -> "FaultClassifier":
        """Train on injector-labelled fingerprints, columns per FEATURE_NAMES."""
        mat = np.asarray(X, dtype=np.float64)
        if mat.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} fingerprint columns, got {mat.shape[1]}")
        order = {fault: k for k, fault in enumerate(FAULT_ORDER)}
        try:
            encoded = np.array([order[label] for label in labels], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown fault class: {exc}") from exc
        self.forest.fit(mat, encoded)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(N, K) distribution over FAULT_ORDER."""
        return self.forest.predict_proba(np.asarray(X, dtype=np.float64))

    def predict_one(self, features: Sequence[float]) -> FaultPrediction:
        """Single fingerprint -> class, confidence, named distribution."""
        row = np.asarray(features, dtype=np.float64).reshape(1, N_FEATURES)
        proba = self.predict_proba(row)[0]
        best = int(np.argmax(proba))
        return FaultPrediction(
            fault_class=FAULT_ORDER[best],
            confidence=float(proba[best]),
            probabilities={fault.value: float(proba[k]) for k, fault in enumerate(FAULT_ORDER)},
        )
