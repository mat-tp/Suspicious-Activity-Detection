"""Classification metrics that stay meaningful when classes are imbalanced.

AD-SVD has 160 pristine videos against about 1,450 for each other imaging condition, so accuracy alone
rewards ignoring the small class. Every report therefore carries macro scores, per-class scores and the
confusion matrix next to accuracy.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score

# These are the scores that are reported with a confidence interval.
HEADLINE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix with rows = true class, columns = predicted class."""
    flat = n_classes * np.asarray(y_true) + np.asarray(y_pred)
    return np.bincount(flat, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def scores_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    """Headline scores from a confusion matrix (fast, so bootstrapping is cheap)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    present = support > 0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str],
                           y_proba: Optional[np.ndarray] = None) -> Dict:
    """All metrics for one set of predictions. Labels are integer class indices."""
    k = len(class_names)
    cm = confusion(y_true, y_pred, k)
    out = scores_from_confusion(cm)
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    per_class = {}
    for i, name in enumerate(class_names):
        p = tp[i] / predicted[i] if predicted[i] else 0.0
        r = tp[i] / support[i] if support[i] else 0.0
        per_class[name] = {
            "precision": float(p), "recall": float(r),
            "f1": float(2 * p * r / (p + r)) if p + r else 0.0,
            "support": int(support[i]),
        }
    out["per_class"] = per_class
    out["confusion_matrix"] = cm.tolist()
    out["confusion_matrix_normalised"] = (
        cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).round(4).tolist()
    out["roc_auc_ovr_macro"] = _auroc(y_true, y_proba, k)
    return out


def _auroc(y_true, y_proba, k) -> Optional[float]:
    """One-vs-rest macro AUROC, or None if probabilities are missing or a class is absent."""
    if y_proba is None or np.ndim(y_proba) != 2 or y_proba.shape[1] != k:
        return None
    if len(np.unique(y_true)) < k:
        return None
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                   average="macro", labels=list(range(k))))
    except ValueError:
        return None


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 groups: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """Percentile bootstrap confidence intervals for the headline metrics.

    Videos are resampled with replacement and the metrics are recomputed each time. With ``groups`` (one id
    per video) whole groups are resampled, because videos of one recording are not independent and
    resampling them one by one gives intervals that are too narrow. Retraining variation is not captured.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    members_by_group = None
    if groups is not None:
        groups = np.asarray(groups)
        members_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = {m: np.empty(n_boot) for m in HEADLINE_METRICS}
    for b in range(n_boot):
        if members_by_group is None:
            idx = rng.integers(0, n, n)
        else:
            picked = rng.integers(0, len(members_by_group), len(members_by_group))
            idx = np.concatenate([members_by_group[g] for g in picked])
        s = scores_from_confusion(confusion(y_true[idx], y_pred[idx], n_classes))
        for m in HEADLINE_METRICS:
            draws[m][b] = s[m]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for m, v in draws.items()}
