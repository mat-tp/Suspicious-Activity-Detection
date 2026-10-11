from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detection.utils import make_bbox
from ..interfaces import BBox, Track
from ..registry import register
from .base import BaseTracker
from .utils import bbox_iou


@register("tracker", "sort")
class SORTTracker(BaseTracker):
    """SORT: Kalman-predicted boxes matched to detections by IoU (Hungarian)."""

    tracker_type = "sort"

    def __init__(self, max_disappeared=30, max_trajectory_len=300, iou_threshold=0.3, min_hits=3):
        super().__init__(max_disappeared, max_trajectory_len)
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.frame_count = 0
        self._kalman_id_counter = 0
        self._predicted: Dict[int, Optional[BBox]] = {}

    class KalmanBoxTracker:
        def __init__(self, bbox: BBox, tracker_id: int):
            self.kf = cv2.KalmanFilter(7, 4)
            self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
            for i in range(3):
                self.kf.transitionMatrix[i, i+4] = 1.0
            self.kf.transitionMatrix[3, 3] = 1.0
            self.kf.measurementMatrix = np.zeros((4, 7), dtype=np.float32)
            for i in range(4):
                self.kf.measurementMatrix[i, i] = 1.0
            self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
            self.kf.processNoiseCov[4:, 4:] *= 0.01
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.3
            self.kf.measurementNoiseCov[2:4, 2:4] *= 0.1
            self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            s = w * h
            r = w / float(h) if h > 0 else 1.0
            self.kf.statePost = np.array([[x], [y], [s], [r], [0], [0], [0]], dtype=np.float32)
            self.time_since_update = 0
            self.id = tracker_id
            self.hits = 1
            self.hit_streak = 1
            self.age = 1

        def predict(self):
            predicted_state = self.kf.predict()
            x, y, s, r = predicted_state[0,0], predicted_state[1,0], predicted_state[2,0], predicted_state[3,0]
            w = np.sqrt(s * r)
            h = s / w if w > 0 else 1
            return make_bbox(int(x - w/2), int(y - h/2), int(w), int(h))

        def update(self, bbox: BBox):
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.age += 1
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            measurement = np.array([[x], [y], [w*h], [w/float(h) if h > 0 else 1.0]], dtype=np.float32)
            self.kf.correct(measurement)

    # ── BaseTracker hooks ─────────────────────────────────────────────────────

    def _begin_frame(self, detections: List[BBox]) -> None:
        """Advance every Kalman filter one step, even on frames with no detections."""
        self.frame_count += 1
        self._predicted = {}
        for tid, track in self.tracks.items():
            try:
                self._predicted[tid] = track["kalman_tracker"].predict()
            except Exception:
                self._predicted[tid] = track.get("bbox", detections[0] if detections else None)

    def _increment_disappeared(self, track_ids: List[int]) -> None:
        super()._increment_disappeared(track_ids)
        for tid in track_ids:
            self.tracks[tid]["kalman_tracker"].time_since_update += 1

    def _start_track(self, bbox: BBox) -> None:
        kalman = self.KalmanBoxTracker(bbox, self._kalman_id_counter)
        self._kalman_id_counter += 1
        self._new_track(bbox["centroid"], bbox, kalman_tracker=kalman, hits=1)

    def _update_matched(self, track: Track, bbox: BBox) -> None:
        track["kalman_tracker"].update(bbox)
        track["hits"] += 1
        self._update_track(track, bbox["centroid"], bbox)

    def _match_and_update(self, detections: List[BBox]) -> None:
        if not self.tracks:
            for det in detections:
                self._start_track(det)
            return

        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, tid in enumerate(track_ids):
            ref = self._predicted.get(tid)
            if ref is None:
                ref = self.tracks[tid].get("bbox")
            if ref is None:
                continue                      # cost stays 0.0 -> never matched
            for j, det in enumerate(detections):
                cost[i, j] = -bbox_iou(ref, det)

        used_rows, used_cols = set(), set()
        for r, c in zip(*linear_sum_assignment(cost)):
            if -cost[r, c] <= self.iou_threshold:
                continue
            self._update_matched(self.tracks[track_ids[r]], detections[c])
            used_rows.add(r)
            used_cols.add(c)

        for j, det in enumerate(detections):
            if j not in used_cols:
                self._start_track(det)
        self._increment_disappeared([tid for i, tid in enumerate(track_ids) if i not in used_rows])

    # ── Tracker interface (differs from the base: only confirmed tracks) ──────

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t["hits"] >= self.min_hits]

    def reset(self) -> None:
        super().reset()
        self.frame_count = 0
        self._kalman_id_counter = 0
        self._predicted = {}
