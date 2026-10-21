"""Motion-based detectors that need no background model: dense optical flow and frame differencing."""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening

from ..detection.utils import STRUCT, _to_gray, make_bbox
from ..interfaces import BBox, Detector
from ..registry import register

# This is the logger of this module.
log = logging.getLogger(__name__)


@register("detector", "dense_flow")
class DenseOpticalFlowDetector(Detector):
    """
    Detects moving regions using Farnebäck dense optical flow magnitude,
    robust to gradual illumination changes (Exposure) and spatial blur (Focus).
    """

    def __init__(
        self,
        magnitude_threshold: float = 2.0,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        adaptive_threshold: bool = True,
    ):
        self.magnitude_threshold = magnitude_threshold
        self.min_contour_area = min_contour_area
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.adaptive_threshold = adaptive_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._mag_history: List[float] = []

    def reset(self) -> None:
        """Forget the previous frame and the flow-magnitude history."""
        self._prev_gray = None
        self._frame_count = 0
        self._mag_history = []

    def _compute_flow_mask(self, gray: np.ndarray) -> np.ndarray:
        # Guard: if shapes differ (e.g. corrupted frame decoded at wrong size),
        # reset prev_gray and return an empty mask rather than crashing.
        if self._prev_gray.shape != gray.shape:
            log.warning(
                "DenseOpticalFlowDetector: shape mismatch prev=%s cur=%s — resetting prev_gray",
                self._prev_gray.shape, gray.shape,
            )
            self._prev_gray = gray
            return np.zeros(gray.shape[:2], dtype=np.uint8)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0,
            )
        except cv2.error as exc:
            log.warning("DenseOpticalFlowDetector: Farneback failed (%s) — skipping frame", exc)
            self._prev_gray = gray
            return np.zeros(gray.shape[:2], dtype=np.uint8)

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Adaptive threshold: use mean + k*std of recent frames
        if self.adaptive_threshold and len(self._mag_history) >= 5:
            mu = float(np.mean(self._mag_history[-20:]))
            sigma = float(np.std(self._mag_history[-20:]))
            threshold = max(self.magnitude_threshold, mu + 0.5 * sigma)
        else:
            threshold = self.magnitude_threshold

        self._mag_history.append(float(magnitude.mean()))

        binary = (magnitude > threshold).astype(np.uint8) * 255
        closed = binary_closing(binary > 0, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        return (opened.astype(np.uint8) * 255)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Boxes of regions whose Farneback flow is above the (adaptive) threshold; ``[]`` for the first frame.
        """
        gray = _to_gray(frame)
        self._frame_count += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        mask = self._compute_flow_mask(gray)
        self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections


@register("detector", "frame_diff")
class AdaptiveFrameDiffDetector(Detector):
    """
    Detects motion via absolute frame difference with adaptive threshold,
    works under heavy blur because it does not maintain a background model.
    """

    def __init__(
        self,
        threshold: int = 25,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        noise_percentile: float = 75.0,
        adaptive: bool = True,
    ):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.noise_percentile = noise_percentile
        self.adaptive = adaptive
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the previous frame."""
        self._prev_gray = None

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Boxes of regions that changed since the previous frame; ``[]`` for the first frame.

        With ``adaptive=True`` the threshold is raised to 1.5x the ``noise_percentile`` of the difference image, so a noisy frame is not read as motion everywhere.
        """
        gray = _to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        # Guard: mismatched shapes from a corrupted frame, reset and skip.
        if self._prev_gray.shape != gray.shape:
            log.warning(
                "AdaptiveFrameDiffDetector: shape mismatch prev=%s cur=%s — resetting",
                self._prev_gray.shape, gray.shape,
            )
            self._prev_gray = gray
            return []

        try:
            diff = cv2.absdiff(self._prev_gray, gray)
        except cv2.error as exc:
            log.warning("AdaptiveFrameDiffDetector: absdiff failed (%s) — skipping frame", exc)
            self._prev_gray = gray
            return []

        self._prev_gray = gray

        if self.adaptive:
            noise_floor = float(np.percentile(diff, self.noise_percentile))
            threshold = max(self.threshold, int(noise_floor * 1.5))
        else:
            threshold = self.threshold

        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        closed = binary_closing(binary > 0, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        mask = (opened.astype(np.uint8) * 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections
