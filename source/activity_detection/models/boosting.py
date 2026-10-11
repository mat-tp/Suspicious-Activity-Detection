from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ..interfaces import Model
from ..models.sklearn_models import SklearnParamsMixin
from ..optional import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, lgb, xgb
from ..registry import register


@register("model", "lightgbm")
class LightGBMModel(SklearnParamsMixin, ClassifierMixin, BaseEstimator, Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31, class_weight="balanced", random_state=42):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state
        self._model = None
        self._scaler = StandardScaler()
        self._le = LabelEncoder()
        self._classes = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled = self._scaler.fit_transform(X)
        self._model = lgb.LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                         learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                         class_weight=self.class_weight, random_state=self.random_state, verbose=-1)
        self._model.fit(X_scaled, y_enc)
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._le.inverse_transform(self._model.predict(self._scaler.transform(X)))

    def predict_proba(self, X):
        if self._model is None:
            return None
        return self._model.predict_proba(self._scaler.transform(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_ if self._model else None

    @property
    def classes_(self):
        return self._classes


@register("model", "xgboost")
class XGBoostModel(SklearnParamsMixin, ClassifierMixin, BaseEstimator, Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 class_weight="balanced", random_state=42):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.random_state = random_state
        self._model = None
        self._scaler = StandardScaler()
        self._le = LabelEncoder()
        self._classes = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled = self._scaler.fit_transform(X)
        # XGBoost has no class_weight option, so "balanced" becomes per-sample weights.
        sample_w = compute_sample_weight("balanced", y_enc) if self.class_weight == "balanced" else None
        self._model = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        learning_rate=self.learning_rate, random_state=self.random_state,
                                        use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
        self._model.fit(X_scaled, y_enc, sample_weight=sample_w)
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._le.inverse_transform(self._model.predict(self._scaler.transform(X)))

    def predict_proba(self, X):
        if self._model is None:
            return None
        return self._model.predict_proba(self._scaler.transform(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_ if self._model else None

    @property
    def classes_(self):
        return self._classes
