"""Pure-NumPy decision tree and random forest (the "custom" forest used as an interpretable baseline).

The split search sorts each candidate feature once and scores every threshold from running
class counts. Labels must be integers 0..K-1 because ``np.bincount`` is used for class counts.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):
    """Entropy-based decision tree grown recursively.

    ``compute_importances`` counts how often a feature is used for a split. It is not an impurity-decrease
    importance, so read it as a rough usage count.

    Args:
        min_samples_split: a node with fewer samples becomes a leaf.
        max_depth: maximum depth.
        n_features: candidate features per split (``None`` = all).
        random_state: seed of the feature sampling.
    """

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.random_state = random_state
        # This is the first node of the tree, built by fit().
        self.root = None
        # This is the number of columns of the training data.
        self.n_total_features = None

    def fit(self, X, y):
        """Grow the tree on feature matrix *X* and integer labels *y*; returns ``self``."""
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
        # This scans every candidate threshold of a feature in one sorted pass, so the cost
        # per feature is O(n log n) instead of O(n * distinct values).
        n = len(y)
        n_classes = int(y.max()) + 1
        parent_entropy = self._entropy(y)
        one_hot = np.eye(n_classes)[y]
        best_gain, best_feat, best_thr = -1.0, None, None
        for feat in feat_idxs:
            col = X[:, feat]
            order = np.argsort(col, kind="stable")
            sorted_col = col[order]
            # This marks the last row of every distinct value, i.e. every threshold "col <= t".
            last_of_value = np.flatnonzero(np.append(sorted_col[1:] != sorted_col[:-1], True))
            left_counts = np.cumsum(one_hot[order], axis=0)[last_of_value]
            n_left = left_counts.sum(axis=1)
            n_right = n - n_left
            right_counts = one_hot.sum(axis=0) - left_counts
            gains = (parent_entropy
                     - (n_left / n) * self._entropy_rows(left_counts)
                     - (n_right / n) * self._entropy_rows(right_counts))
            # A threshold that leaves one side empty (the largest value) is not a split.
            gains[n_right < 1] = 0.0
            i = int(np.argmax(gains))
            if gains[i] > best_gain:
                best_gain, best_feat, best_thr = float(gains[i]), feat, sorted_col[last_of_value[i]]
        return best_feat, best_thr

    @staticmethod
    def _entropy_rows(counts):
        # This is the entropy of every row of a class-count matrix (empty rows give 0).
        totals = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
        logs = np.log(probs, out=np.zeros_like(probs), where=probs > 0)
        return -(probs * logs).sum(axis=1)

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
        """Per-feature split counts summed over the tree (length *n_features*, not normalised)."""
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
    """Bagged :class:`DecisionTree` ensemble with majority voting.

    Args:
        n_trees: number of trees.
        max_depth: depth limit (``None`` = unlimited).
        min_samples_split: see :class:`DecisionTree`.
        n_features: candidate features per split (default ``sqrt(n_features)``).
        class_weight: ``"balanced"`` draws the bootstrap samples with probability inversely proportional
            to class frequency; ``None`` is a plain bootstrap.
        random_state: seed.
        n_jobs: processes that grow the trees (``-1`` = all cores). The samples are drawn first, so the
            forest is the same for any value.

    ``feature_importances_`` is the normalised split-usage count over all trees.
    """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None, class_weight=None, random_state=42,
                 n_jobs=1):
        self.n_trees = n_trees
        # No depth limit is stored as a very large number.
        self.max_depth = max_depth or 1_000_000
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        # These are the fitted trees.
        self.trees = []
        # This is how often each feature is used for a split, as a share of all splits.
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit ``n_trees`` trees on bootstrap resamples of (*X*, *y*); returns ``self``."""
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
        # The random draws happen here, in order, so that the parallel and the serial
        # forest are identical.
        draws = []
        for _ in range(self.n_trees):
            seed = int(rng.integers(0, 1_000_000))
            idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True, p=sample_weights)
            draws.append((seed, idxs))
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._grow_tree)(X[idxs], y[idxs], seed, n_feat) for seed, idxs in draws
        )
        raw = np.sum([t.compute_importances(X.shape[1]) for t in self.trees], axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw
        return self

    def _grow_tree(self, X, y, seed, n_feat):
        tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                            n_features=n_feat, random_state=seed)
        return tree.fit(X, y)

    def predict(self, X):
        """Predicted label per row of *X*: the most frequent vote of the trees."""
        all_preds = np.array([t.predict(X) for t in self.trees])
        # This counts the votes per class for every row; ties go to the smaller label.
        votes = np.apply_along_axis(np.bincount, 0, all_preds.astype(int),
                                    minlength=int(all_preds.max()) + 1)
        return np.argmax(votes, axis=0)
