"""Tests for imbalance handling."""
from collections import Counter

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from activity_detection.config import load_config
from activity_detection.evaluation.cross_validation import cross_validate_model
from activity_detection.models.imbalance import (
    RESAMPLING,
    ImbalanceConfig,
    ResamplingClassifier,
    build_model,
    resample,
)


def imbalanced(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], [30, 200, 180])
    return rng.normal(size=(len(y), 6)) + 0.4 * y[:, None], y


@pytest.mark.parametrize("method", [m for m in RESAMPLING if m not in ("none", "smote_enn", "smote_tomek")])
def test_full_ratio_balances_classes(method):
    X, y = imbalanced()
    _, y_new = resample(X, y, method, 1.0, 5, {}, 0)
    counts = Counter(y_new.tolist())
    assert len(set(counts.values())) == 1


def test_partial_ratio_only_raises_the_smallest_class():
    X, y = imbalanced()
    _, y_new = resample(X, y, "smote", 0.5, 5, {}, 0)
    counts = Counter(y_new.tolist())
    assert counts[0] == 100 and counts[1] == 200 and counts[2] == 180


def test_none_leaves_data_untouched():
    X, y = imbalanced()
    X_new, y_new = resample(X, y, "none", 1.0, 5, {}, 0)
    assert X_new is X and y_new is y


def test_resampling_happens_in_fit_only():
    X, y = imbalanced()
    model = ResamplingClassifier(RandomForestClassifier(10, random_state=0), "smote")
    model.fit(X, y)
    assert model.n_train_resampled_ > model.n_train_original_
    assert len(model.predict(X[:7])) == 7          # prediction input is never resampled


def test_synthetic_samples_stay_inside_their_own_class_range():
    X, y = imbalanced()
    X_new, y_new = resample(X, y, "mixup", 1.0, 5, {}, 0)
    original = X[y == 0]
    synthetic = X_new[len(y):][y_new[len(y):] == 0]
    assert len(synthetic) == 170
    assert (synthetic >= original.min(axis=0) - 1e-9).all() and (synthetic <= original.max(axis=0) + 1e-9).all()


def test_tiny_class_falls_back_instead_of_crashing():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40 + [2])
    X = rng.normal(size=(81, 4))
    _, y_new = resample(X, y, "smote", 1.0, 5, {}, 0)
    assert Counter(y_new.tolist())[2] == 40


def test_test_folds_keep_their_original_size_under_resampling():
    X, y = imbalanced()
    model = ResamplingClassifier(RandomForestClassifier(10, random_state=0), "smote")
    cv = cross_validate_model(model, X, y, ["a", "b", "c"], n_splits=4, n_boot=20)
    assert len(cv["_oof_pred"]) == len(y)
    assert sum(c["support"] for c in cv["pooled"]["per_class"].values()) == len(y)


def test_class_weight_setting_reaches_the_models():
    off = ImbalanceConfig(class_weight="none")
    on = ImbalanceConfig(class_weight="balanced")
    assert build_model("random_forest_sklearn", {}, off, 0).class_weight is None
    assert build_model("random_forest_sklearn", {}, on, 0).class_weight == "balanced"
    assert build_model("svm_rbf", {}, off, 0).class_weight is None
    # a class_weight written on the model itself wins over the global one
    assert build_model("svm_rbf", {"class_weight": "balanced"}, off, 0).class_weight == "balanced"


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        ImbalanceConfig(resampling="smoot")
    with pytest.raises(ValueError):
        ImbalanceConfig(class_weight="sometimes")
    with pytest.raises(ValueError):
        ImbalanceConfig(sampling_ratio=0)


def test_config_block_is_read_from_yaml():
    cfg = load_config("group_a_classical", overrides={"imbalance": {"resampling": "adasyn", "sampling_ratio": 0.5}})
    parsed = ImbalanceConfig.from_config(cfg.get("imbalance"))
    assert parsed.resampling == "adasyn" and parsed.sampling_ratio == 0.5 and parsed.class_weight == "balanced"
