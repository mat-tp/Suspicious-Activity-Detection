"""IoU-based tracker with Hungarian (linear sum) assignment."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..interfaces import BBox
from ..registry import register
from .base import BaseTracker
from .utils import bbox_iou


@register("tracker", "iou_hungarian")
class IoUHungarianTracker(BaseTracker):
    """Optimal IoU matching via the Hungarian algorithm.

    Prefers this over centroid distance because bounding-box overlap
    is more stable than raw centroid proximity when detections vary in
    size across frames.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_trajectory_len: int = 300,
        iou_threshold: float = 0.3,
        keep_finished_tracks: bool = True,
    ) -> None:
        """Args:
            iou_threshold: a track and a detection match only if their IoU is above this. A track keeps its
                last box (there is no motion prediction), so an object that moves so fast that consecutive
                boxes overlap less than this starts a new track.
            max_disappeared, max_trajectory_len, keep_finished_tracks: see ``BaseTracker``.
        """
        super().__init__(max_disappeared, max_trajectory_len, keep_finished_tracks)
        self.iou_threshold = iou_threshold

    def _match_and_update(self, detections: List[BBox]) -> None:
        """Hungarian assignment on negative IoU; unmatched detections start new tracks."""
        if not self.tracks:
            for det in detections:
                self._new_track(det["centroid"], bbox=det)
            return

        track_ids = list(self.tracks.keys())
        n_tracks, n_dets = len(track_ids), len(detections)

        # Cost matrix: negative IoU (minimising the cost maximises the IoU).
        cost = np.zeros((n_tracks, n_dets), dtype=float)
        for i, tid in enumerate(track_ids):
            track_bbox = self.tracks[tid].get("bbox")
            if track_bbox is None:
                continue
            for j, det in enumerate(detections):
                cost[i, j] = -bbox_iou(track_bbox, det)

        row_idx, col_idx = linear_sum_assignment(cost)

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if -cost[r, c] <= self.iou_threshold:
                continue  # IoU too low — treat as unmatched
            self._update_track(
                self.tracks[track_ids[r]],
                detections[c]["centroid"],
                bbox=detections[c],
            )
            used_rows.add(r)
            used_cols.add(c)

        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det["centroid"], bbox=det)

        unmatched_tracks = [
            track_ids[i] for i in range(n_tracks) if i not in used_rows
        ]
        self._increment_disappeared(unmatched_tracks)
