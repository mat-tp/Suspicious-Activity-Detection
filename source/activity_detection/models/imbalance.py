"""Class-imbalance handling, chosen in the ``imbalance:`` section of a config.

``class_weight`` is ``balanced`` (rarer classes count more) or ``none``. ``resampling`` changes the
training set (oversampling, undersampling or noise augmentation) and is applied inside each training
fold only, never to a test fold. ``sampling_ratio`` says how far to balance, from 0 to 1. All methods
work on feature vectors; they do not augment videos.
"""
from __future__ import annotations

import inspect
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler

from ..registry import build, get_class

# imbalanced-learn is optional, so a missing package is reported when a resampling method is used.
try:
    from imblearn import combine, over_sampling, under_sampling
except ImportError:
    combine = over_sampling = under_sampling = None

# This is the logger of this module.
log = logging.getLogger(__name__)

# These are the accepted values of the class-weight option.
CLASS_WEIGHTS = ("balanced", "none")
# These are the accepted values of the resampling option.
RESAMPLING = ("none", "random_over", "smote", "borderline_smote", "adasyn",
              "smote_tomek", "smote_enn", "random_under", "gaussian_noise", "mixup")
# These are the resampling methods that add synthetic or repeated minority samples.
OVERSAMPLERS = {"random_over", "smote", "borderline_smote", "adasyn",
                "smote_tomek", "smote_enn", "gaussian_noise", "mixup"}


@dataclass
class ImbalanceConfig:
    """How class imbalance is handled (the ``imbalance:`` block of a config).

    Attributes:
        class_weight: ``"balanced"`` or ``"none"``.
        resampling: method applied to the training fold only.
        sampling_ratio: how far to balance, in (0, 1].
        k_neighbors: neighbours used by the SMOTE-style methods.
        params: extra options of a method, such as ``noise_std``.
    """

    class_weight: str = "balanced"
    resampling: str = "none"
    sampling_ratio: float = 1.0
    k_neighbors: int = 5
    params: Dict[str, Any] = field(default_factory=dict)   # extra options, e.g. noise_std

    def __post_init__(self):
        if self.class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"imbalance.class_weight must be one of {CLASS_WEIGHTS}, got {self.class_weight!r}")
        if self.resampling not in RESAMPLING:
            raise ValueError(f"imbalance.resampling must be one of {RESAMPLING}, got {self.resampling!r}")
        if not 0 < self.sampling_ratio <= 1:
            raise ValueError("imbalance.sampling_ratio must be in (0, 1]")

    @classmethod
    def from_config(cls, section) -> "ImbalanceConfig":
        """Build from the config's `imbalance:` block (missing block = defaults)."""
        if section is None:
            return cls()
        data = section.to_dict() if hasattr(section, "to_dict") else dict(section)
        return cls(**data)

    def describe(self) -> str:
        """One-line human-readable summary printed at the start of a run."""
        text = f"class_weight={self.class_weight}, resampling={self.resampling}"
        if self.resampling != "none":
            text += f" (ratio {self.sampling_ratio}, k={self.k_neighbors})"
        return text


def build_model(name: str, params: Dict[str, Any], imbalance: ImbalanceConfig, seed: int):
    """Create a registered model with the imbalance settings applied."""
    params = dict(params or {})
    accepts = inspect.signature(get_class("model", name).__init__).parameters
    if "class_weight" in accepts:
        # a model's own class_weight in the config wins over the global setting
        params.setdefault("class_weight", None if imbalance.class_weight == "none" else "balanced")
    elif imbalance.class_weight == "balanced":
        log.info("%s has no class weights; relying on resampling (%s) only", name, imbalance.resampling)

    model = build("model", name, random_state=seed, **params)
    if imbalance.resampling == "none":
        return model
    return ResamplingClassifier(
        model, resampling=imbalance.resampling, sampling_ratio=imbalance.sampling_ratio,
        k_neighbors=imbalance.k_neighbors, resampling_params=imbalance.params, random_state=seed)


class ResamplingClassifier(BaseEstimator, ClassifierMixin):
    """Resample the training data inside fit(), then fit the wrapped model.

    Features are standardised before resampling so that distance-based methods
    (SMOTE and friends) are not dominated by large-valued features, then mapped
    back before being passed to the model.
    """

    def __init__(self, estimator, resampling: str = "smote", sampling_ratio: float = 1.0,
                 k_neighbors: int = 5, resampling_params: Optional[dict] = None,
                 random_state: int = 42):
        self.estimator = estimator
        self.resampling = resampling
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.resampling_params = resampling_params
        self.random_state = random_state

    def fit(self, X, y):
        """Standardise, resample the training data, and fit a clone of the wrapped estimator.

        Resampling happens here, inside ``fit``, so it only ever touches the training fold and
        never the validation fold; returns ``self``.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        scaler = StandardScaler().fit(X)
        X_new, y_new = resample(scaler.transform(X), y, self.resampling, self.sampling_ratio,
                                self.k_neighbors, self.resampling_params or {}, self.random_state)
        self.estimator_ = clone(self.estimator).fit(scaler.inverse_transform(X_new), y_new)
        self.classes_ = np.unique(y)
        self.n_train_original_ = len(y)
        self.n_train_resampled_ = len(y_new)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return getattr(self.estimator_, "feature_importances_", None)


def target_counts(counts: Counter, resampling: str, ratio: float) -> Dict[int, int]:
    """Desired sample count for each class that has to change."""
    if resampling in OVERSAMPLERS:
        goal = int(round(ratio * max(counts.values())))
        return {c: goal for c, n in counts.items() if n < goal}
    goal = int(round(min(counts.values()) / ratio))
    return {c: goal for c, n in counts.items() if n > goal}


def resample(X, y, method: str, ratio: float, k_neighbors: int, params: dict, seed: int):
    """Return resampled (X, y). Called with training data only."""
    if method == "none":
        return X, y
    counts = Counter(y.tolist())
    strategy = target_counts(counts, method, ratio)
    if not strategy:
        return X, y

    if method == "gaussian_noise":
        return _add_synthetic(X, y, strategy, seed, _noisy_copy, params.get("noise_std", 0.1))
    if method == "mixup":
        return _add_synthetic(X, y, strategy, seed, _mix_pair, params.get("alpha", 0.4))

    smallest = min(counts.values())
    if method in {"smote", "borderline_smote", "adasyn", "smote_tomek", "smote_enn"} and smallest < 2:
        log.warning("A class has only %d sample(s); SMOTE needs at least 2. Using random_over.", smallest)
        method = "random_over"
    k = max(1, min(k_neighbors, smallest - 1))
    try:
        return _imblearn_sampler(method, strategy, k, seed).fit_resample(X, y)
    except ValueError as err:
        if method != "adasyn":
            raise
        # ADASYN makes no samples when the minority class has no hard-to-learn region
        log.warning("ADASYN generated no samples (%s). Using smote instead.", err)
        return _imblearn_sampler("smote", strategy, k, seed).fit_resample(X, y)


def _imblearn_sampler(method: str, strategy: Dict[int, int], k: int, seed: int):
    if over_sampling is None:
        raise ImportError("This resampling method needs imbalanced-learn: "
                          "pip install imbalanced-learn")
    kwargs = dict(sampling_strategy=strategy, random_state=seed)
    if method == "random_over":
        return over_sampling.RandomOverSampler(**kwargs)
    if method == "random_under":
        return under_sampling.RandomUnderSampler(**kwargs)
    if method == "smote":
        return over_sampling.SMOTE(k_neighbors=k, **kwargs)
    if method == "borderline_smote":
        return over_sampling.BorderlineSMOTE(k_neighbors=k, m_neighbors=max(k, 2), **kwargs)
    if method == "adasyn":
        return over_sampling.ADASYN(n_neighbors=k, **kwargs)
    smote = over_sampling.SMOTE(k_neighbors=k, sampling_strategy=strategy, random_state=seed)
    if method == "smote_tomek":
        return combine.SMOTETomek(smote=smote, random_state=seed)
    if method == "smote_enn":
        return combine.SMOTEENN(smote=smote, random_state=seed)
    raise ValueError(f"Unknown resampling method: {method}")


def _noisy_copy(rows: np.ndarray, rng, noise_std: float) -> np.ndarray:
    """One new sample: a random existing row plus noise scaled to the class spread."""
    base = rows[rng.integers(len(rows))]
    spread = rows.std(axis=0) if len(rows) > 1 else np.ones(rows.shape[1])
    return base + rng.normal(0.0, noise_std, rows.shape[1]) * spread


def _mix_pair(rows: np.ndarray, rng, alpha: float) -> np.ndarray:
    """One new sample: a blend of two random rows of the same class."""
    a, b = rows[rng.integers(len(rows))], rows[rng.integers(len(rows))]
    lam = rng.beta(alpha, alpha)
    return lam * a + (1 - lam) * b


def _add_synthetic(X, y, strategy, seed, make_one, option):
    """Append synthetic samples until each class in `strategy` reaches its target."""
    rng = np.random.default_rng(seed)
    new_X, new_y = [X], [y]
    for cls, goal in strategy.items():
        rows = X[y == cls]
        extra = np.array([make_one(rows, rng, option) for _ in range(goal - len(rows))])
        new_X.append(extra)
        new_y.append(np.full(len(extra), cls, dtype=y.dtype))
    return np.vstack(new_X), np.concatenate(new_y)
