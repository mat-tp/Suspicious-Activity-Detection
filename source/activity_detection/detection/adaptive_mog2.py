"""Confidence-driven adaptive MOG2 detector.

Plain MOG2 uses one fixed variance threshold. Under blur or bad exposure that
threshold often gives an empty or noisy mask. This detector watches its own
output and relaxes the threshold only when the output looks bad.

For every frame:

1. Segment with the normal threshold and score the mask (0 to 1).
2. Blend that score with a running "track quality" from the tracker:
       combined = w * mask_quality + (1 - w) * track_quality
3. If combined >= q, keep the normal result.
4. Otherwise lower the threshold in proportion to how bad the score is,
   segment again, and keep the new mask only if it scores higher.

The background model is updated exactly once per frame, by the normal call.
The relaxed segmentation is a second look at the same frame with learning rate
0, so it does not update the model. Because of this, frames that are not
adapted give exactly the same result as plain MOG2. Set update_once=False to
get the older behaviour, where the second segmentation also updated the model.

Parameter symbols:
    q    quality_threshold        w    mask_weight
    tau_r  relaxed_var_threshold  tau_min  threshold_floor
"""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..interfaces import BBox, Track
from ..registry import register
from .mog2 import MOG2Detector
from .utils import make_bbox

# Mask-quality scoring constants (see _score_mask)
COVERAGE_LOW = 0.0005    # below this fraction of foreground pixels the mask is empty
COVERAGE_HIGH = 0.25     # above this the background model has collapsed
COVERAGE_TARGET = 0.02   # coverage that scores best
FRAG_MAX = 30            # more blobs than this counts as fragmented
BLOB_SIZE_REL = 0.002    # median blob area / frame area that scores well


@register("detector", "adaptive_mog2")
class AdaptiveMOG2Detector(MOG2Detector):

    def __init__(self, history: int = 20, var_threshold: float = 16,
                 detect_shadows: bool = True,
                 morph_close_iter: int = 2, morph_open_iter: int = 1,
                 min_contour_area: int = 500,
                 relaxed_var_threshold: float = 10.0,   # tau_r
                 quality_threshold: float = 0.4,        # q
                 mask_weight: float = 0.7,              # w
                 threshold_floor: float = 8.0,          # tau_min
                 min_track_len: int = 5,
                 ema_alpha: float = 0.8,
                 update_once: bool = True,
                 log_decisions: bool = False):
        super().__init__(history=history, var_threshold=var_threshold,
                         detect_shadows=detect_shadows,
                         morph_close_iter=morph_close_iter,
                         morph_open_iter=morph_open_iter,
                         min_contour_area=min_contour_area)
        self.relaxed_var_threshold = relaxed_var_threshold
        self.quality_threshold = quality_threshold
        self.mask_weight = float(np.clip(mask_weight, 0.0, 1.0))
        self.threshold_floor = threshold_floor
        self.min_track_len = max(min_track_len, 1)
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.update_once = update_once
        self.log_decisions = log_decisions
        self._start_video()

    # ---- per-video state --------------------------------------------------

    def _start_video(self) -> None:
        self._track_quality = 1.0           # optimistic until the tracker reports
        self._frames_seen = 0
        self._totals: Dict[str, float] = dict(
            frames=0, triggered=0, accepted=0,
            mask_quality=0.0, track_quality=0.0, threshold=0.0)
        self.decisions: List[Dict[str, Any]] = []   # filled only if log_decisions

    def reset(self) -> None:
        super().reset()
        self._start_video()

    def get_diagnostics(self) -> Dict[str, float]:
        """Summary of what the detector did on the current video."""
        n = max(self._totals["frames"], 1)
        triggered = max(self._totals["triggered"], 1)
        return {
            "avg_mask_quality": self._totals["mask_quality"] / n,
            "avg_track_quality": self._totals["track_quality"] / n,
            "avg_threshold_used": self._totals["threshold"] / n,
            "adaptive_trigger_rate": self._totals["triggered"] / n,
            "rerun_accept_rate": self._totals["accepted"] / triggered,
            "n_frames": float(self._totals["frames"]),
        }

    # ---- tracker feedback -------------------------------------------------

    def feedback(self, tracks: List[Track]) -> None:
        """Update the track-quality estimate. Call after tracker.update().

        raw = (mean track length / min_track_len, capped at 1)
              * (share of tracks matched in this frame)

        An exponential moving average smooths it, so one bad frame does not
        swing the detector and cause a feedback loop.
        """
        if tracks:
            lengths = [len(t["trajectory"]) for t in tracks]
            length_score = min(float(np.mean(lengths)) / self.min_track_len, 1.0)
            matched = sum(1 for t in tracks if t.get("disappeared", 0) == 0)
            raw = length_score * matched / len(tracks)
        else:
            raw = 0.0
        self._track_quality = (self.ema_alpha * self._track_quality
                               + (1.0 - self.ema_alpha) * raw)

    # ---- mask scoring -----------------------------------------------------

    def _score_mask(self, mask: np.ndarray, contours) -> float:
        """Score a foreground mask from 0 (useless) to 1 (good).

        Three scores are multiplied, so one bad aspect sinks the whole score:
        coverage (how much of the frame is foreground), fragmentation (number
        of blobs) and blob size (median blob area relative to the frame).
        """
        coverage = float(mask.sum() / 255) / mask.size
        if coverage < COVERAGE_LOW or coverage > COVERAGE_HIGH:
            return 0.0

        areas = [cv2.contourArea(c) for c in contours]
        areas = [a for a in areas if a >= self.min_contour_area]
        if not areas:
            return 0.0

        coverage_score = np.clip(1.0 - abs(coverage - COVERAGE_TARGET) / 0.23, 0.0, 1.0)
        frag_score = np.clip(1.0 - len(areas) / FRAG_MAX, 0.0, 1.0)
        size_score = np.tanh(np.median(areas) / mask.size / BLOB_SIZE_REL)
        return float(coverage_score * frag_score * size_score)

    # ---- segmentation -----------------------------------------------------

    def _segment(self, frame: np.ndarray, threshold: float, learning_rate: float):
        """Run MOG2 once and return (mask, boxes)."""
        self.mog2.setVarThreshold(threshold)
        mask = self.clean_mask(self.mog2.apply(frame, learningRate=learning_rate))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append(make_bbox(x, y, w, h))
        return mask, contours, boxes

    def _relaxed_threshold(self, combined: float) -> float:
        """Lower the threshold more the worse the combined score is."""
        deficit = 1.0 - combined / self.quality_threshold      # 0..1
        t = self.var_threshold - (self.var_threshold - self.relaxed_var_threshold) * deficit
        return float(np.clip(t, self.threshold_floor, self.var_threshold))

    def detect(self, frame: np.ndarray) -> List[BBox]:
        # Step 1: the normal MOG2 call. This is the one and only model update,
        # so the nominal mask is exactly what plain MOG2 would produce.
        mask, contours, boxes = self._segment(frame, self.var_threshold, -1.0)
        mask_q = self._score_mask(mask, contours)
        combined = (self.mask_weight * mask_q
                    + (1.0 - self.mask_weight) * self._track_quality)

        used_threshold = self.var_threshold
        relaxed_t, relaxed_q, accepted = None, None, False

        # OpenCV sets the learning rate from its own count of apply() calls in
        # the first history/2 frames, so no extra calls are made until then.
        self._frames_seen += 1
        warmed_up = self._frames_seen > self.history // 2

        if warmed_up and combined < self.quality_threshold:
            # Step 2: look at the same frame with a lower threshold.
            # Learning rate 0 means this call does not learn from the frame.
            # (update_once=False restores the old behaviour: a second learning call.)
            relaxed_t = self._relaxed_threshold(combined)
            rate = 0.0 if self.update_once else -1.0
            r_mask, r_contours, r_boxes = self._segment(frame, relaxed_t, rate)
            relaxed_q = self._score_mask(r_mask, r_contours)
            if relaxed_q > mask_q:                 # keep only if strictly better
                boxes, used_threshold, accepted = r_boxes, relaxed_t, True

        self.mog2.setVarThreshold(self.var_threshold)
        self._record(mask_q, combined, used_threshold, relaxed_t, relaxed_q, accepted)
        return boxes

    def _record(self, mask_q, combined, used_threshold, relaxed_t, relaxed_q, accepted):
        t = self._totals
        t["frames"] += 1
        t["mask_quality"] += mask_q
        t["track_quality"] += self._track_quality
        t["threshold"] += used_threshold
        if relaxed_t is not None:
            t["triggered"] += 1
            t["accepted"] += int(accepted)
        if self.log_decisions:
            self.decisions.append(dict(
                frame=int(t["frames"]) - 1, mask_quality=mask_q,
                track_quality=self._track_quality, combined=combined,
                relaxed_threshold=relaxed_t, relaxed_quality=relaxed_q,
                accepted=accepted))
