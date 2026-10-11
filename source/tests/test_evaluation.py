"""Tests for metrics and cross-validation."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from activity_detection.evaluation.cross_validation import (
    check_training_viability,
    cross_validate_model,
)
from activity_detection.evaluation.metrics import bootstrap_ci, classification_metrics

NAMES = ["a", "b", "c"]


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    m = classification_metrics(y, pred, NAMES)
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(m["weighted_f1"], f1_score(y, pred, average="weighted"))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, pred))
    assert sum(c["support"] for c in m["per_class"].values()) == 300


def test_accuracy_hides_a_ignored_minority_class_but_macro_f1_does_not():
    y = np.array([0] * 95 + [1] * 5)
    pred = np.zeros(100, dtype=int)                 # never predicts class 1
    m = classification_metrics(y, pred, ["big", "small"])
    assert m["accuracy"] == 0.95
    assert m["per_class"]["small"]["recall"] == 0.0
    assert m["macro_f1"] < 0.5


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 400)
    pred = np.where(rng.random(400) < 0.8, y, rng.integers(0, 3, 400))
    ci = bootstrap_ci(y, pred, 3, n_boot=300)
    point = classification_metrics(y, pred, NAMES)
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert ci[name][0] <= point[name] <= ci[name][1]


def test_cross_validation_predicts_every_video_once_and_is_repeatable():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1, 2], [20, 60, 60])
    X = rng.normal(size=(len(y), 5)) + y[:, None]
    model = KNeighborsClassifier(3)
    a = cross_validate_model(model, X, y, NAMES, n_splits=4, n_boot=50)
    b = cross_validate_model(model, X, y, NAMES, n_splits=4, n_boot=50)
    assert len(a["_oof_pred"]) == len(y)
    assert a["pooled"]["macro_f1"] == b["pooled"]["macro_f1"]
    assert len(a["fold_scores"]["macro_f1"]["values"]) == 4


def test_repeated_cv_reports_each_repeat():
    rng = np.random.default_rng(4)
    y = np.repeat([0, 1], 40)
    X = rng.normal(size=(80, 4)) + y[:, None]
    r = cross_validate_model(KNeighborsClassifier(3), X, y, ["a", "b"], n_splits=4, n_repeats=3, n_boot=20)
    assert len(r["repeats"]["macro_f1"]) == 3
    assert len(r["fold_scores"]["accuracy"]["values"]) == 12


def test_viability_check_rejects_tiny_classes():
    ok, _ = check_training_viability(np.array([0] * 30 + [1] * 30))
    bad, msg = check_training_viability(np.array([0] * 30 + [1]))
    assert ok and not bad and "fewer than" in msg
