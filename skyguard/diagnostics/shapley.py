"""Layer 6b — Shapley explanations for the fault classifier.

Why Shapley, and why only here: the autoencoder already explains itself via
per-variable reconstruction errors — no post-hoc method needed. The forest,
though, votes across 14 fingerprint features, and an operator deserves to know
*which evidence* carried the verdict. Shapley values split the prediction
fairly across features: they sum to f(point) − f(background), so the ranked
list is a complete account, not a highlight reel.

Exact enumeration up to `exact_max_features`, permutation sampling above it
(KernelSHAP-style budget), over an interventional background drawn from normal
data — the reference is "a normal observation", which is exactly the contrast
an operator wants.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

import numpy as np

from ..config import DEFAULT_CONFIG, ShapleyConfig

ScoreFunction = Callable[[np.ndarray], np.ndarray]


def shapley_values(
    point: np.ndarray,
    background: np.ndarray,
    predict_fn: ScoreFunction,
    config: ShapleyConfig | None = None,
) -> np.ndarray:
    """Attribute f(point) − f(background) across features.

    `predict_fn` maps (M, F) to (M,) scores — typically the predicted class's
    probability. Deterministic given the config seed.
    """
    cfg = config or DEFAULT_CONFIG.shapley
    vec = np.asarray(point, dtype=np.float64).ravel()
    ref = _background_reference(background)
    if vec.shape != ref.shape:
        raise ValueError(f"point dim {vec.shape} != background dim {ref.shape}")
    n_features = vec.shape[0]
    if n_features == 0:
        raise ValueError("cannot explain a zero-feature point")
    if n_features <= cfg.exact_max_features:
        return _exact(vec, ref, predict_fn)
    return _sampled(vec, ref, predict_fn, cfg)


def rank_contributions(
    feature_names: Sequence[str], values: np.ndarray, top_k: int | None = None
) -> list:
    """(name, value) pairs sorted by |contribution|, operators read top-down."""
    order = np.argsort(-np.abs(np.asarray(values, dtype=np.float64)))
    pairs = [(feature_names[i], float(values[i])) for i in order]
    return pairs if top_k is None else pairs[:top_k]


def _background_reference(background: np.ndarray) -> np.ndarray:
    """Interventional reference: the mean normal observation.

    A single mean reference (not the full distribution) keeps exact mode
    tractable and matches the sampling estimator's baseline.
    """
    mat = np.asarray(background, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError("background must be a non-empty (N, F) matrix")
    if not np.all(np.isfinite(mat)):
        raise ValueError("background must be finite")
    capped = mat[: max(mat.shape[0], 1)]
    return capped.mean(axis=0)


def _masked_matrix(vec: np.ndarray, ref: np.ndarray, present: np.ndarray) -> np.ndarray:
    """One row per subset mask: present features from the point, rest reference."""
    return np.where(np.asarray(present, dtype=bool), vec, ref)


def _exact(vec: np.ndarray, ref: np.ndarray, predict_fn: ScoreFunction) -> np.ndarray:
    """Exact Shapley via all 2^F coalitions against the mean reference."""
    n_features = vec.shape[0]
    masks = np.array(list(itertools.product([False, True], repeat=n_features)), dtype=bool)
    scores = np.asarray(predict_fn(_masked_matrix(vec, ref, masks)), dtype=np.float64).ravel()
    index_of = {mask: k for k, mask in enumerate(map(tuple, masks.tolist()))}
    phi = np.zeros(n_features, dtype=np.float64)
    denom = math.factorial(n_features)
    for i in range(n_features):
        for mask, base_score in zip(masks, scores):
            if mask[i]:
                continue
            with_i = list(mask)
            with_i[i] = True
            marginal = scores[index_of[tuple(with_i)]] - base_score
            weight = math.factorial(int(mask.sum())) * math.factorial(n_features - int(mask.sum()) - 1) / denom
            phi[i] += weight * marginal
    return phi


def _sampled(
    vec: np.ndarray, ref: np.ndarray, predict_fn: ScoreFunction, config: ShapleyConfig
) -> np.ndarray:
    """Permutation-sampling Shapley: random feature orderings, marginal gains.

    Each sample draws one ordering and credits every feature its marginal
    contribution as it joins — the mean over orderings converges to the
    Shapley value, and the budget keeps it affordable past exact range.
    """
    rng = np.random.default_rng(config.seed)
    n_features = vec.shape[0]
    phi = np.zeros(n_features, dtype=np.float64)
    for _ in range(config.n_samples):
        order = rng.permutation(n_features)
        present = np.zeros(n_features, dtype=bool)
        baseline = float(np.asarray(predict_fn(ref.reshape(1, -1)), dtype=np.float64).ravel()[0])
        running = baseline
        current = ref.copy()
        for feat in order:
            current[int(feat)] = vec[int(feat)]
            present[int(feat)] = True
            nxt = float(np.asarray(predict_fn(current.reshape(1, -1)), dtype=np.float64).ravel()[0])
            phi[int(feat)] += nxt - running
            running = nxt
    return phi / config.n_samples
