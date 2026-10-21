"""Cascaded detector: MOG2 first, then dense optical flow, then frame differencing.

The switch is decided from the number of boxes seen in a warm-up window. It is a fallback for videos
where the background model finds (almost) nothing, not a learned selector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from ..detection.mog2 import MOG2Detector
from ..detection.motion import AdaptiveFrameDiffDetector, DenseOpticalFlowDetector
from ..interfaces import BBox, Detector
from ..registry import register

# This is the logger of this module.
log = logging.getLogger(__name__)


@register("detector", "cascaded_fallback")
class CascadedFallbackDetector(Detector):
    """
    Tries MOG2 first; if insufficient detections in warmup, falls back to
    dense optical flow; if that fails, falls back to frame differencing.
    Guarantees maximum detection coverage without ground truth.
    """

    def __init__(
        self,
        mog2_history: int = 20,
        mog2_var_threshold: float = 16,
        mog2_detect_shadows: bool = True,
        flow_magnitude_threshold: float = 2.0,
        flow_adaptive_threshold: bool = True,
        diff_threshold: int = 25,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        warmup_frames: int = 10,
        min_detections_per_warmup: int = 3,
    ):
        # This is the first stage: MOG2 with the simple adaptive threshold.
        self._mog2 = MOG2Detector(
            history=mog2_history,
            var_threshold=mog2_var_threshold,
            detect_shadows=mog2_detect_shadows,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
            min_contour_area=min_contour_area,
            adaptive=True,  # enable adaptive within MOG2
        )
        # This is the second stage, used when MOG2 finds almost nothing.
        self._flow = DenseOpticalFlowDetector(
            magnitude_threshold=flow_magnitude_threshold,
            min_contour_area=min_contour_area,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
            adaptive_threshold=flow_adaptive_threshold,
        )
        # This is the last stage, used when the flow detector also finds almost nothing.
        self._diff = AdaptiveFrameDiffDetector(
            threshold=diff_threshold,
            min_contour_area=min_contour_area,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
        )

        self.warmup_frames = warmup_frames
        self.min_detections_per_warmup = min_detections_per_warmup

        self._frame_count: int = 0
        self._active_detector: str = "mog2"
        self._warmup_detection_count: int = 0
        self._detector_switches: List[Dict] = []

    def reset(self) -> None:
        """Reset all three detectors and go back to MOG2 for a new video."""
        self._mog2.reset()
        self._flow.reset()
        self._diff.reset()
        self._frame_count = 0
        self._active_detector = "mog2"
        self._warmup_detection_count = 0
        self._detector_switches = []

    def _try_fallback(self, reason: str) -> None:
        switch = {
            "frame": self._frame_count,
            "from":  self._active_detector,
            "reason": reason,
        }
        if self._active_detector == "mog2":
            self._active_detector = "dense_flow"
            log.info("Cascaded detector: MOG2 → Dense Flow at frame %d (%s)", self._frame_count, reason)
        elif self._active_detector == "dense_flow":
            self._active_detector = "frame_diff"
            log.info("Cascaded detector: Dense Flow → Frame Diff at frame %d (%s)", self._frame_count, reason)
        switch["to"] = self._active_detector
        self._detector_switches.append(switch)
        self._warmup_detection_count = 0

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Boxes from the detector that is currently active.

        MOG2 always runs, so its background model stays warm. If it finds fewer than ``min_detections_per_warmup``
        boxes in the warm-up window, dense flow takes over, and if flow also finds too few, frame differencing
        takes over for the rest of the video.
        """
        self._frame_count += 1

        # Always feed MOG2 so its background model stays warm
        mog2_dets = self._mog2.detect(frame)

        if self._active_detector == "mog2":
            self._warmup_detection_count += len(mog2_dets)
            if self._frame_count == self.warmup_frames:
                if self._warmup_detection_count < self.min_detections_per_warmup:
                    self._try_fallback("insufficient MOG2 detections in warmup")
                else:
                    return mog2_dets
            return mog2_dets

        if self._active_detector == "dense_flow":
            flow_dets = self._flow.detect(frame)
            self._warmup_detection_count += len(flow_dets)
            if self._frame_count == self.warmup_frames * 2:
                if self._warmup_detection_count < self.min_detections_per_warmup:
                    self._try_fallback("insufficient Dense Flow detections")
            return flow_dets

        # Final fallback: frame differencing
        return self._diff.detect(frame)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Which detector is active and when each switch happened (for the detector statistics report)."""
        return {
            "active_detector":    self._active_detector,
            "detector_switches":  self._detector_switches,
            "n_switches":         len(self._detector_switches),
        }
