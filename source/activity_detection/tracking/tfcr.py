"""Tracker that matches boxes by IoU and keeps a small online appearance model per track.

This is a lightweight take on the target-focusing idea of Yuan et al. (2020), not a copy of their
convolutional tracker. It manages its own track lifecycle and does not derive from ``BaseTracker``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detection.utils import make_bbox
from ..interfaces import BBox, Track, Tracker
from ..registry import register
from ..tracking.utils import bbox_iou


@register("tracker", "tfcr")
class TFCRTracker(Tracker):
    """IoU + Hungarian matching with a per-track ridge-regression model on HOG features.

    The model is updated every frame with a target-focusing loss,
    ``J(w) = ||w.phi - y||^2 - eta ||w.phi||^2 + lambda ||w||^2``, which widens the gap between the
    target and the background response (Yuan et al., 2020). HOG replaces the VGG features of the
    original so that the tracker runs on a CPU.

    Args:
        max_disappeared: frames a track survives without a match before it is retired.
        max_trajectory_len: number of trajectory points kept per track.
        iou_threshold: smallest IoU for a detection to be matched to a track.
        eta: strength of the target-focusing term.
        lam: ridge weight of the model.
        lr: learning rate of the online update.
        scale_factors: box scales tried when the box size is estimated.
        hog_cell: HOG cell size in pixels.
    """

    # This is the default strength of the target-focusing term (eta in the loss).
    DEFAULT_ETA   = 0.5
    # This is the default ridge weight of the appearance model (lambda in the loss).
    DEFAULT_LAM   = 1e-4
    # This is the default learning rate of the online update.
    DEFAULT_LR    = 0.02
    # These are the default box scales that are tried when the size of a box is estimated.
    DEFAULT_SCALES = [0.95, 1.0, 1.05]

    def __init__(
        self,
        max_disappeared: int = 30,
        max_trajectory_len: int = 300,
        iou_threshold: float = 0.3,
        eta: float = DEFAULT_ETA,
        lam: float = DEFAULT_LAM,
        lr: float = DEFAULT_LR,
        scale_factors: Optional[List[float]] = None,
        hog_cell: int = 8,
        keep_finished_tracks: bool = True,
    ):
        self.keep_finished_tracks = keep_finished_tracks   # archive retired tracks (see BaseTracker)
        self.finished: List[Track] = []
        self.max_disappeared    = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.iou_threshold      = iou_threshold
        self.eta                = eta          # target-focusing coefficient
        self.lam                = lam          # ridge regularisation weight
        self.lr                 = lr           # online model update learning rate
        self.scale_factors      = scale_factors or self.DEFAULT_SCALES
        self.hog_cell           = hog_cell

        # This is the id that the next new track gets.
        self.next_id    = 0
        # This is where the live tracks are kept, with the track id as the key.
        self.tracks: Dict[int, Track]       = {}
        self.pruned_count: int              = 0
        # Per-track appearance models: dict of track id to weight vector w
        self._models: Dict[int, np.ndarray] = {}

    def _extract_hog(self, frame_gray: np.ndarray, bbox: BBox) -> np.ndarray:
        """HOG descriptor of the box region (zeros if the crop is too small or there is no frame)."""
        if frame_gray is None:
            # Fallback: return a unit-norm dummy feature so the model update
            # step still executes (prevents NaN in ridge regression).
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        x, y, w, h = (
            max(0, bbox["x"]), max(0, bbox["y"]),
            max(1, bbox["w"]), max(1, bbox["h"]),
        )
        # Guard: ensure crop coordinates stay within the frame
        fh, fw = frame_gray.shape[:2]
        x2 = min(x + w, fw)
        y2 = min(y + h, fh)
        crop = frame_gray[y:y2, x:x2]

        if crop.size == 0:
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        # Resize to a fixed patch size for consistent feature dimensionality.
        # 64 by 64 was chosen to match the TFCR training patch convention (Yuan
        # et al. 2020, section 4.1: training patch is 5 times the target in width and
        # height; HOG at 8px cells gives 8 by 8 = 64 bins).
        patch_size = max(self.hog_cell * 8, 16)
        try:
            patch = cv2.resize(crop, (patch_size, patch_size))
        except cv2.error:
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        # Sobel gradients for HOG
        gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Compute a 9-bin HOG descriptor over the full patch
        n_bins = 9
        hist, _ = np.histogram(
            ang.flatten(), bins=n_bins, range=(0, 2 * np.pi),
            weights=mag.flatten(),
        )
        hist_norm = hist / (np.linalg.norm(hist) + 1e-7)
        return hist_norm.astype(np.float32)

    def _init_model(self, feat: np.ndarray) -> np.ndarray:
        """Random small-variance Gaussian weights for a new track's appearance model."""
        rng = np.random.default_rng()
        return (rng.standard_normal(feat.shape) * 0.01).astype(np.float32)

    def _update_model(
        self,
        track_id: int,
        feat: np.ndarray,
        iou_score: float,
    ) -> None:
        """One gradient step on the target-focusing loss.

        ``dw = -lr * (2 (1 - eta) phi (phi.w) - 2 phi y + 2 lambda w)``. The *iou_score* scales the step, so
        frames where the prediction and the detection agree move the model more.
        """
        if track_id not in self._models:
            self._models[track_id] = self._init_model(feat)

        w      = self._models[track_id]
        y_des  = 1.0   # desired response for the target sample
        phi    = feat

        # Response at current weight
        y_pred = float(np.dot(phi, w))

        # Target-focusing gradient (Eq. 5, Yuan et al. 2020)
        # dJ/dy' = 2[(1 - eta)y' - y]
        # Gradient w.r.t. w via the chain rule: dJ/dw = (dJ/dy') * phi + 2 * lambda * w
        grad_response = 2.0 * ((1.0 - self.eta) * y_pred - y_des)
        grad = grad_response * phi + 2.0 * self.lam * w

        # IoU-weighted learning rate: high-confidence matches update more
        effective_lr = self.lr * (0.5 + 0.5 * iou_score)

        self._models[track_id] = w - effective_lr * grad

    def _estimate_scale(
        self,
        frame_gray: Optional[np.ndarray],
        bbox: BBox,
        candidate_detections: List[BBox],
    ) -> BBox:
        """The candidate scale with the highest model response (1.0 if there is no model yet)."""
        track_id = bbox.get("id")
        if track_id is None or track_id not in self._models or frame_gray is None:
            # No appearance model yet, return unchanged scale
            return bbox

        best_score  = -np.inf
        best_bbox   = bbox
        cx, cy      = bbox["cx"], bbox["cy"]

        for sf in self.scale_factors:
            # Scale bounding box around its centroid
            nw = max(4, int(bbox["w"] * sf))
            nh = max(4, int(bbox["h"] * sf))
            nx = max(0, cx - nw // 2)
            ny = max(0, cy - nh // 2)
            cand = make_bbox(nx, ny, nw, nh)

            feat  = self._extract_hog(frame_gray, cand)
            score = float(np.dot(feat, self._models[track_id]))

            if score > best_score:
                best_score = score
                best_bbox  = cand

        return best_bbox

    def _new_track(self, bbox: BBox, frame_gray: Optional[np.ndarray] = None) -> None:
        """
        Initialise a new track with an appearance model.

        Corresponds to TFCR §3.3 — Model initialisation: the appearance model
        w is randomly initialised; the training patch is larger than the target
        (by the scale factors) to include context / background samples.
        """
        tid = self.next_id
        self.tracks[tid] = {
            "id":           tid,
            "centroid":     bbox["centroid"],
            "bbox":         bbox,
            "trajectory":   [bbox["centroid"]],
            "disappeared":  0,
            "tracker_type": "tfcr",
        }
        # Initialise appearance model for this track
        feat = self._extract_hog(frame_gray, bbox)
        self._models[tid] = self._init_model(feat)
        self.next_id += 1

    def _update_track(
        self,
        track: Track,
        bbox: BBox,
        iou_score: float,
        frame_gray: Optional[np.ndarray] = None,
    ) -> None:
        """Store the matched detection in the track and update its appearance model (every frame)."""
        tid = track["id"]
        # Scale estimation: select optimal scale candidate
        bbox = self._estimate_scale(frame_gray, {**bbox, "id": tid}, [])

        track["centroid"]    = bbox["centroid"]
        track["bbox"]        = bbox
        track["disappeared"] = 0
        track["trajectory"].append(bbox["centroid"])

        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len:]

        # Online appearance model update (target-focusing gradient step)
        feat = self._extract_hog(frame_gray, bbox)
        self._update_model(tid, feat, iou_score)

    def update(
        self,
        detections: List[BBox],
        frame_gray: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Match detections to tracks (IoU + Hungarian), then update the appearance models.

        Passing no *frame_gray* skips the HOG step, and the tracker then works from boxes alone.
        """
        if not detections:
            for t in self.tracks.values():
                t["disappeared"] += 1
            self._prune_expired()
            return list(self.tracks.values())

        if not self.tracks:
            for d in detections:
                self._new_track(d, frame_gray)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, tid in enumerate(track_ids):
            track_bbox = self.tracks[tid].get("bbox")
            if track_bbox is None:
                cost[i, :] = 0.0
                continue
            for j, det in enumerate(detections):
                cost[i, j] = -bbox_iou(track_bbox, det)

        row_idx, col_idx = linear_sum_assignment(cost)

        used_rows: set = set()
        used_cols: set = set()

        for r, c in zip(row_idx, col_idx):
            iou_val = -cost[r, c]
            if iou_val <= self.iou_threshold:
                continue   # reject weak match
            self._update_track(
                self.tracks[track_ids[r]],
                detections[c],
                iou_score=iou_val,
                frame_gray=frame_gray,
            )
            used_rows.add(r)
            used_cols.add(c)

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det, frame_gray)

        # Increment disappeared counter for unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1

        self._prune_expired()
        return list(self.tracks.values())

    def _prune_expired(self) -> None:
        """Remove tracks that have been unmatched for too long."""
        expired = [
            tid for tid, t in self.tracks.items()
            if t["disappeared"] > self.max_disappeared
        ]
        for tid in expired:
            track = self.tracks.pop(tid)
            if self.keep_finished_tracks:
                self.finished.append(track)
            self._models.pop(tid, None)   # free appearance model memory
            self.pruned_count += 1

    def get_active_tracks(self) -> List[Track]:
        """Tracks that are alive right now."""
        return list(self.tracks.values())

    def get_all_tracks(self) -> List[Track]:
        """Retired tracks followed by live tracks (see ``BaseTracker.get_all_tracks``)."""
        return list(self.finished) + self.get_active_tracks()

    def reset(self) -> None:
        """Reset all track and appearance-model state for a new video."""
        self.tracks.clear()
        self.finished = []
        self._models.clear()
        self.next_id      = 0
        self.pruned_count = 0
