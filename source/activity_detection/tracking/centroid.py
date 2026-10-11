"""Greedy nearest-neighbour centroid tracker."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial.distance import cdist

from ..interfaces import BBox, Track
from ..registry import register
from .base import BaseTracker


@register("tracker", "centroid")
class CentroidTracker(BaseTracker):
    """Greedy nearest-neighbour matching on centroid Euclidean distance."""

    def _match_and_update(self, detections: List[BBox]) -> None:
        centroids = [d["centroid"] for d in detections]

        if not self.tracks:
            for c in centroids:
                self._new_track(c)
            return

        track_ids = list(self.tracks.keys())
        old_cents = np.array(
            [self.tracks[tid]["centroid"] for tid in track_ids], dtype=float
        )
        new_cents = np.array(centroids, dtype=float)
        dist_matrix = cdist(old_cents, new_cents, metric="euclidean")

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        rows, cols = np.unravel_index(
            np.argsort(dist_matrix, axis=None), dist_matrix.shape
        )
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            self._update_track(self.tracks[track_ids[r]], centroids[c])
            used_rows.add(r)
            used_cols.add(c)

        for j, c in enumerate(centroids):
            if j not in used_cols:
                self._new_track(c)

        unmatched_tracks = [
            track_ids[i] for i in range(len(track_ids)) if i not in used_rows
        ]
        self._increment_disappeared(unmatched_tracks)
