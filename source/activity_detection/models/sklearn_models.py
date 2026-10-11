from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from ..interfaces import Model
from ..models.decision_tree import RandomForestCustom
from ..registry import register


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


class SklearnParamsMixin:
    """Mixin that adds sklearn-compatible get_params / set_params via stored instance attrs."""

    def get_params(self, deep: bool = True) -> dict:
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != "self" and hasattr(self, k)}

    def set_params(self, **params):
        current = self.get_params()
        current.update(params)
        self.__init__(**current)
        return self


class SklearnModelAdapter(ClassifierMixin, BaseEstimator, Model):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        try:
            return self.pipeline.predict_proba(X)
        except Exception:
            return None

    @property
    def feature_importances_(self):
        try:
            return self.pipeline.named_steps["clf"].feature_importances_
        except Exception:
            return None

    @property
    def classes_(self):
        try:
            return self.pipeline.named_steps["clf"].classes_
        except Exception:
            return None


@register("model", "random_forest_sklearn")
class RandomForestSklearnModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_estimators=200, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(_build_pipeline(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                                class_weight=class_weight, random_state=random_state, n_jobs=n_jobs)))


@register("model", "random_forest_custom")
class RandomForestCustomModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_trees=200, max_depth=None, min_samples_split=2, class_weight="balanced", random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(_build_pipeline(RandomForestCustom(n_trees=n_trees, max_depth=max_depth,
                                                            min_samples_split=min_samples_split, class_weight=class_weight,
                                                            random_state=random_state)))


@register("model", "svm_rbf")
class SVMRBFModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, C=10, gamma="scale", class_weight="balanced", random_state=42):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(_build_pipeline(SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
                                             random_state=random_state, probability=True)))


@register("model", "knn")
class KNNModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_neighbors=5, metric="euclidean", **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        super().__init__(_build_pipeline(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)))


@register("model", "mlp")
class MLPModel(SklearnParamsMixin, SklearnModelAdapter):
    """Small fully connected network on the handcrafted feature vector.

    The neural-network counterpart to the tree models: same inputs, same
    cross-validation, no raw video. Features are standardised first. Early stopping
    holds out 10% of the training data; on very small datasets (a few hundred
    videos) set early_stopping: false. scikit-learn's
    MLP has no class_weight option, so "balanced" is applied as per-sample weights.
    """

    def __init__(self, hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                 max_iter=300, early_stopping=True, n_iter_no_change=20,
                 class_weight="balanced", random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(_build_pipeline(MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes), alpha=alpha,
            learning_rate_init=learning_rate_init, max_iter=max_iter,
            early_stopping=early_stopping, n_iter_no_change=n_iter_no_change,
            random_state=random_state)))

    def fit(self, X, y):
        weights = compute_sample_weight("balanced", y) if self.class_weight == "balanced" else None
        self.pipeline.fit(X, y, clf__sample_weight=weights)
        return self
