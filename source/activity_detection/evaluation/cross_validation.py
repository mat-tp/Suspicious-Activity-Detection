"""Stratified (optionally repeated) cross-validation.

Everything the model learns, including class weights, resampling and feature
scaling, happens inside fit() on the training folds only. Test folds are never
resampled or augmented.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .metrics import (
    HEADLINE_METRICS,
    bootstrap_ci,
    classification_metrics,
    confusion,
    scores_from_confusion,
)

# This is the logger of this module.
log = logging.getLogger(__name__)


def check_training_viability(y: np.ndarray, n_folds: int = 5,
                             min_per_class: int = 3) -> Tuple[bool, str]:
    """Check there is enough data for cross-validation."""
    counts = Counter(y)
    n_samples, n_classes = len(y), len(counts)

    if n_samples < n_classes * min_per_class:
        return False, (f"Only {n_samples} samples for {n_classes} classes "
                       f"(need at least {n_classes * min_per_class})")
    small = [c for c, n in counts.items() if n < min_per_class]
    if small:
        return False, f"Classes with fewer than {min_per_class} samples: {small}"
    if n_samples < n_folds * n_classes:
        return False, (f"{n_folds}-fold CV needs at least {n_folds * n_classes} samples, "
                       f"have {n_samples}")
    return True, "OK"


def _predict_proba(model, X) -> Optional[np.ndarray]:
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def _split(X, y, groups, n_splits: int, seed: int):
    """Stratified folds; with groups, no group is ever split across train and test."""
    if groups is None:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(X, y, groups))


def _run_fold(model, X, y, train_idx, test_idx, n_classes: int) -> Dict[str, Any]:
    fitted = clone(model).fit(X[train_idx], y[train_idx])
    pred = fitted.predict(X[test_idx])
    proba = _predict_proba(fitted, X[test_idx])
    if proba is not None and proba.shape[1] != n_classes:
        proba = None     # a class was missing from this training fold
    train_acc = float((fitted.predict(X[train_idx]) == y[train_idx]).mean())
    return {"test_idx": test_idx, "pred": pred, "proba": proba, "train_accuracy": train_acc}


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, class_names: List[str],
                         n_splits: int = 5, n_repeats: int = 1, seed: int = 42,
                         n_jobs: int = 1, n_boot: int = 1000,
                         groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Cross-validate one model and return a JSON-friendly summary.

    *y* holds class indices 0..K-1 that match *class_names*. *groups* (optional) is one id per sample;
    samples with the same id always land in the same fold.

    The result has ``fold_scores`` (per-fold scores of every repeat), ``pooled`` (metrics on the out-of-fold
    predictions of the first repeat, where each video is predicted once by a model that never saw it),
    ``ci95`` (bootstrap intervals of the pooled metrics) and ``repeats`` (macro-F1 per repeat).
    """
    k = len(class_names)
    fold_rows: List[Dict[str, Any]] = []
    pooled_pred: List[np.ndarray] = []
    pooled_proba: List[Optional[np.ndarray]] = []

    for repeat in range(n_repeats):
        splits = _split(X, y, groups, n_splits, seed + repeat)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_fold)(model, X, y, tr, te, k) for tr, te in splits)

        oof_pred = np.empty(len(y), dtype=int)
        oof_proba = np.full((len(y), k), np.nan)
        for fold, res in enumerate(results):
            te = res["test_idx"]
            oof_pred[te] = res["pred"]
            if res["proba"] is not None:
                oof_proba[te] = res["proba"]
            scores = scores_from_confusion(confusion(y[te], res["pred"], k))
            scores.update(repeat=repeat, fold=fold, train_accuracy=res["train_accuracy"])
            fold_rows.append(scores)
        pooled_pred.append(oof_pred)
        pooled_proba.append(None if np.isnan(oof_proba).any() else oof_proba)

    summary: Dict[str, Any] = {"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed,
                               "grouped": groups is not None}
    summary["fold_scores"] = {
        m: {"mean": float(np.mean([r[m] for r in fold_rows])),
            "std": float(np.std([r[m] for r in fold_rows])),
            "values": [r[m] for r in fold_rows]}
        for m in (*HEADLINE_METRICS, "weighted_f1", "train_accuracy")
    }
    summary["overfit_gap"] = (summary["fold_scores"]["train_accuracy"]["mean"]
                              - summary["fold_scores"]["accuracy"]["mean"])
    summary["pooled"] = classification_metrics(y, pooled_pred[0], class_names, pooled_proba[0])
    summary["ci95"] = bootstrap_ci(y, pooled_pred[0], k, n_boot=n_boot, seed=seed, groups=groups)
    if n_repeats > 1:
        per_repeat = [scores_from_confusion(confusion(y, p, k))["macro_f1"] for p in pooled_pred]
        summary["repeats"] = {"macro_f1": per_repeat,
                              "macro_f1_mean": float(np.mean(per_repeat)),
                              "macro_f1_std": float(np.std(per_repeat))}
    summary["_oof_pred"] = pooled_pred[0]        # for ensembles; removed before saving
    return summary
