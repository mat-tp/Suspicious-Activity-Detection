"""Per-video feature aggregation.

Each track produces a ``FeatureDict`` of scalar statistics.  The
``FeatureAggregator`` pools these into a single fixed-length vector by
applying one or more reduction functions (mean, max, std) across all
tracks for each feature key.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..interfaces import FeatureDict

# Aggregation functions available by name.
_AGG_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(np.mean(a)),
    "max": lambda a: float(np.max(a)),
    "min": lambda a: float(np.min(a)),
    "std": lambda a: float(np.std(a)) if len(a) > 1 else 0.0,
    "median": lambda a: float(np.median(a)),
}


class FeatureAggregator:
    """Reduce a list of per-track feature dicts to one per-video feature dict."""

    def __init__(self, aggregation_fns: List[str] | None = None) -> None:
        requested = aggregation_fns or ["mean", "max", "std"]
        unknown = set(requested) - _AGG_FNS.keys()
        if unknown:
            raise ValueError(
                f"Unknown aggregation function(s): {unknown}. "
                f"Supported: {list(_AGG_FNS)}"
            )
        self.aggregation_fns: List[str] = requested
        # These are the functions that pool the per-track values, in the order of the
        # requested names.
        self._fns = [_AGG_FNS[name] for name in requested]

    def aggregate(self, track_features: List[FeatureDict]) -> Optional[FeatureDict]:
        """Return a single feature dict, or ``None`` if *track_features* is empty."""
        if not track_features:
            return None

        # Collect all keys present across all tracks.
        all_keys: set[str] = set()
        for tf in track_features:
            all_keys.update(tf.keys())

        video_fv: FeatureDict = {"n_valid_tracks": float(len(track_features))}

        for key in sorted(all_keys):
            values = [tf[key] for tf in track_features if key in tf]
            if not values:
                continue
            arr = np.array(values, dtype=float)
            for name, fn in zip(self.aggregation_fns, self._fns):
                video_fv[f"{name}_{key}"] = fn(arr)

        return video_fv
