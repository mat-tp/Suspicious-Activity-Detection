from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.random_state = random_state
        self.root = None
        self.n_total_features = None

    def fit(self, X, y):
        self.n_total_features = X.shape[1]
        self.n_split_features = (X.shape[1] if not self.n_features else min(X.shape[1], self.n_features))
        rng = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, depth=0, rng=rng)
        return self

    def _grow(self, X, y, depth, rng):
        if X.shape[0] == 0:
            return {"leaf": True, "value": 0}
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or X.shape[0] < self.min_samples_split):
            return {"leaf": True, "value": self._majority(y)}
        feat_idxs = rng.choice(self.n_total_features, self.n_split_features, replace=False)
        best_feat, best_thr = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return {"leaf": True, "value": self._majority(y)}
        left_mask = X[:, best_feat] <= best_thr
        return {"leaf": False, "feature": best_feat, "threshold": best_thr,
                "left": self._grow(X[left_mask], y[left_mask], depth + 1, rng),
                "right": self._grow(X[~left_mask], y[~left_mask], depth + 1, rng)}

    def _best_split(self, X, y, feat_idxs):
        best_gain, best_feat, best_thr = -1.0, None, None
        for feat in feat_idxs:
            col = X[:, feat]
            for thr in np.unique(col):
                gain = self._info_gain(y, col, thr)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def _info_gain(self, y, col, thr):
        left_mask = col <= thr
        if left_mask.sum() < 1 or (~left_mask).sum() < 1:
            return 0.0
        n = len(y)
        n_l = left_mask.sum()
        return (self._entropy(y) - (n_l / n) * self._entropy(y[left_mask]) - ((n - n_l) / n) * self._entropy(y[~left_mask]))

    def _entropy(self, y):
        counts = np.bincount(y)
        ps = counts / len(y)
        return -np.sum(ps[ps > 0] * np.log(ps[ps > 0]))

    def _majority(self, y):
        return int(np.bincount(y).argmax()) if len(y) > 0 else 0

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])

    def compute_importances(self, n_features):
        importances = np.zeros(n_features)
        self._accumulate(self.root, importances)
        return importances

    def _accumulate(self, node, importances):
        if node["leaf"]:
            return
        importances[node["feature"]] += 1.0
        self._accumulate(node["left"], importances)
        self._accumulate(node["right"], importances)


class RandomForestCustom(BaseEstimator):
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None, class_weight=None, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth or 1_000_000
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_feat = self.n_features or int(np.sqrt(X.shape[1]))
        self.classes_ = np.unique(y)
        self.trees = []
        sample_weights = None
        if self.class_weight == "balanced":
            class_counts = np.bincount(y)
            class_w = 1.0 / class_counts
            sample_weights = class_w[y]
            sample_weights /= sample_weights.sum()
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=n_feat, random_state=int(rng.integers(0, 1_000_000)))
            if sample_weights is not None:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True, p=sample_weights)
            else:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        raw = np.sum([t.compute_importances(X.shape[1]) for t in self.trees], axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw
        return self

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        result = []
        for i in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)
