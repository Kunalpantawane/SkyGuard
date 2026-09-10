"""Evaluation metrics for anomaly-injected data (`context/evaluation.md`).

Accuracy is reported but never headlined: on 99.5 % normal data, "always
normal" scores 99.5 %. The operational numbers are false-alarm rate (a QC
system that cries wolf gets switched off), event recall with detection delay
(what an operator experiences), per-class recall (drift is hard, spikes are
easy), genuine-extreme false alarms (the flagship: real weather must not be
flagged), and calibration (a 90 % confidence must mean 90 %).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# numpy 2.0 renamed trapz to trapezoid and deprecated the old spelling; the
# repo pins no numpy version, so bind whichever this install actually has.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Point-level precision/recall/F1/false-alarm-rate plus supports."""
    true = np.asarray(y_true, dtype=bool).ravel()
    pred = np.asarray(y_pred, dtype=bool).ravel()
    if true.shape != pred.shape:
        raise ValueError("truth and predictions must share their shape")
    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    tn = int(np.sum(~true & ~pred))
    n_normal = int(np.sum(~true))
    return {
        "accuracy": (tp + tn) / max(true.size, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_alarm_rate": fp / max(n_normal, 1),
        "n_true": float(tp + fn),
        "n_pred": float(tp + fp),
    }


def event_recall(
    spans: Sequence[Tuple[int, int]], y_pred: np.ndarray
) -> Dict[str, object]:
    """Event level: a fault span counts as detected if any point fires inside.

    Point recall over-rewards long faults; event recall is what an operator
    experiences. Delay = intervals from fault onset to first alarm.
    """
    pred = np.asarray(y_pred, dtype=bool).ravel()
    detected = 0
    delays: List[float] = []
    for start, end in spans:
        if start < 0 or end >= pred.size or end < start:
            raise ValueError(f"span ({start}, {end}) is out of range")
        hits = np.flatnonzero(pred[start : end + 1])
        if hits.size:
            detected += 1
            delays.append(float(hits[0]))
    return {
        "event_recall": detected / max(len(spans), 1),
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "n_spans": float(len(spans)),
        "n_detected": float(detected),
    }


def spans_from_labels(y_true: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs as (start, end) spans for event scoring."""
    true = np.asarray(y_true, dtype=bool).ravel()
    spans: List[Tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(true):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(true) - 1))
    return spans


def per_class_recall(
    class_values: np.ndarray, y_pred: np.ndarray,
    background: Sequence[str] = ("none", "genuine_extreme"),
) -> Dict[str, float]:
    """Recall per fault class. Aggregate numbers hide that drift is hard and
    spikes are easy — this breakdown is where that truth comes out."""
    classes = np.asarray(class_values, dtype=object).ravel()
    pred = np.asarray(y_pred, dtype=bool).ravel()
    if classes.shape != pred.shape:
        raise ValueError("class labels and predictions must share their shape")
    out: Dict[str, float] = {}
    for name in sorted(set(classes.tolist()) - set(background)):
        mask = classes == name
        out[str(name)] = float(np.sum(pred[mask]) / max(int(np.sum(mask)), 1))
    return out


def genuine_extreme_far(class_values: np.ndarray, y_pred: np.ndarray) -> float:
    """Flagship number: fraction of real extreme-weather points wrongly flagged."""
    classes = np.asarray(class_values, dtype=object).ravel()
    pred = np.asarray(y_pred, dtype=bool).ravel()
    mask = classes == "genuine_extreme"
    if not np.any(mask):
        return 0.0
    return float(np.sum(pred[mask]) / int(np.sum(mask)))


def confusion_matrix(
    true_labels: Sequence[str], pred_labels: Sequence[str], label_order: Sequence[str]
) -> np.ndarray:
    """Fault-class confusion, rows = truth, columns = prediction."""
    order = list(label_order)
    index = {name: k for k, name in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        if true in index and pred in index:
            matrix[index[true], index[pred]] += 1
    return matrix


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, object]:
    """ECE: do 90 %-confident observations come out anomalous ~90 % of the
    time? A confidence nobody can trust is decoration."""
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    if probs.shape != truth.shape:
        raise ValueError("probabilities and labels must share their shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_prob = float(np.mean(probs[mask]))
        mean_truth = float(np.mean(truth[mask]))
        ece += (count / probs.size) * abs(mean_prob - mean_truth)
        table.append({"bin_mid": (lo + hi) / 2, "mean_prob": mean_prob,
                      "mean_truth": mean_truth, "n": float(count)})
    return {"ece": float(ece), "bins": table}


def _threshold_sweep(scores: np.ndarray, labels: np.ndarray, max_points: int = 200):
    """Shared sweep for ROC and PR: descending unique cuts, thinned for JSON."""
    order = np.argsort(-scores)
    ranked_truth = labels[order]
    tp = np.cumsum(ranked_truth)
    fp = np.cumsum(~ranked_truth)
    keep = np.ones(scores.size, dtype=bool)
    if scores.size > 1:
        # One point per distinct score. Keeping duplicates would put several
        # points at the same cut and bend the trapezoid areas.
        keep[:-1] = scores[order][:-1] != scores[order][1:]
    tp, fp = tp[keep], fp[keep]
    if tp.size > max_points:
        idx = np.unique(np.linspace(0, tp.size - 1, max_points).astype(int))
        tp, fp = tp[idx], fp[idx]
    return tp.astype(np.float64), fp.astype(np.float64)


def roc_curve(scores: np.ndarray, labels: np.ndarray,
              max_points: int = 200) -> Dict[str, object]:
    """ROC over continuous scores, with the trapezoidal area under it.

    Thresholded F1 answers "how good is this cut". AUC answers "how good is the
    ranking, whatever cut you choose", which is what a deployment retuning its
    own threshold actually needs to know.
    """
    probs = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    if probs.shape != truth.shape:
        raise ValueError("scores and labels must share their shape")
    n_pos, n_neg = int(np.sum(truth)), int(np.sum(~truth))
    if n_pos == 0 or n_neg == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan")}
    tp, fp = _threshold_sweep(probs, truth, max_points)
    tpr = np.concatenate(([0.0], tp / n_pos, [1.0]))
    fpr = np.concatenate(([0.0], fp / n_neg, [1.0]))
    return {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr],
            "auc": float(_trapezoid(tpr, fpr))}


def pr_curve(scores: np.ndarray, labels: np.ndarray,
             max_points: int = 200) -> Dict[str, object]:
    """Precision-recall curve plus average precision.

    On data that is 97 % normal, PR is the honest picture and ROC flatters:
    a large false-positive count barely moves FPR but wrecks precision, and
    precision is what decides whether an operator keeps the system switched on.
    """
    probs = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    if probs.shape != truth.shape:
        raise ValueError("scores and labels must share their shape")
    n_pos = int(np.sum(truth))
    if n_pos == 0:
        return {"recall": [0.0, 1.0], "precision": [0.0, 0.0],
                "average_precision": float("nan"), "baseline": 0.0}
    tp, fp = _threshold_sweep(probs, truth, max_points)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1.0)
    average_precision = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    return {"recall": [float(v) for v in recall],
            "precision": [float(v) for v in precision],
            "average_precision": average_precision,
            "baseline": float(n_pos / probs.size)}


def score_histogram(scores: np.ndarray, labels: np.ndarray,
                    n_bins: int = 20) -> Dict[str, object]:
    """Score distribution split by truth: the separation the detector achieved.

    Overlap between the two histograms is the part no threshold can fix.
    """
    probs = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    normal, _ = np.histogram(probs[~truth], bins=edges)
    anomalous, _ = np.histogram(probs[truth], bins=edges)
    return {"edges": [float(v) for v in edges],
            "normal": [int(v) for v in normal],
            "anomalous": [int(v) for v in anomalous]}
