from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.ndimage import binary_closing

from ..detection.utils import STRUCT, _to_gray, clean_binary_mask, make_bbox
from ..interfaces import BBox, Detector
from ..registry import register


@register("detector", "mog2")
class MOG2Detector(Detector):
    """
    MOG2 background subtraction.

    Shadow handling (tuning pass)
    ------------------------------
    Previously `detect_shadows` defaulted to False, so OpenCV's shadow
    model never ran and cast shadows were classified as solid foreground
    -- a common source of oversized/merged bounding boxes in outdoor or
    side-lit footage. This version:

      * defaults `detect_shadows=True` so MOG2's own shadow model runs,
      * explicitly separates the mask values MOG2 can produce
        (0 = background, 127 = shadow, 255 = sure foreground) instead of
        `raw_mask > 0`, which treated shadow pixels as foreground and
        silently defeated the shadow model even when it was enabled,
      * exposes `shadow_as_foreground_weight` so shadow pixels can be kept
        only where they adjoin sure foreground (the likely dark side of a
        real object) rather than either "always foreground" or "always
        discarded".

    var_threshold default (tuning pass)
    ------------------------------------
    Previously defaulted to 25, an unexplained deviation from the
    literature. `var_threshold=16` is the value derived analytically in
    the algorithm's originating work (Zivkovic & van der Heijden, 2006,
    "Efficient Adaptive Density Estimation per Image Pixel for the Task
    of Background Subtraction") as 4-sigma, and is the value documented
    by OpenCV itself as the typical setting. This is now the default here
    and throughout the configs. If a project-specific value other than 16
    is ever adopted, it should be backed by a sensitivity sweep on the
    actual dataset (see scripts/mog2_threshold_sweep.py) rather than
    asserted, and the justification should be recorded alongside the
    config that sets it.
    """
    def __init__(self, history=20, var_threshold=16, detect_shadows=True,
                 shadow_value=127, shadow_as_foreground_weight=0.0,
                 morph_close_iter=2, morph_open_iter=1, min_contour_area=500,
                 adaptive=False):   # NEW: adaptive flag
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.shadow_value = shadow_value
        # 0.0 -> shadows fully excluded from foreground (cleanest boxes)
        # 1.0 -> shadows treated exactly like sure foreground (old behaviour)
        # (0,1) -> shadows kept only where adjoining sure foreground
        self.shadow_as_foreground_weight = float(np.clip(shadow_as_foreground_weight, 0.0, 1.0))
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.min_contour_area = min_contour_area
        self.adaptive = adaptive
        self._make_mog2()

    def _make_mog2(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows)

    def clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        sure_fg = raw_mask >= 200  # 255 in OpenCV's shadow-detection convention
        if self.detect_shadows and self.shadow_as_foreground_weight > 0.0:
            shadow = np.isclose(raw_mask, self.shadow_value, atol=20) & ~sure_fg
            if self.shadow_as_foreground_weight >= 1.0:
                binary = sure_fg | shadow
            else:
                sure_fg_dilated = binary_closing(sure_fg, structure=STRUCT, iterations=1)
                binary = sure_fg | (shadow & sure_fg_dilated)
        else:
            binary = sure_fg
        return clean_binary_mask(binary, self.morph_close_iter, self.morph_open_iter)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        # Adaptive threshold adjustment if enabled
        if self.adaptive:
            gray = _to_gray(frame)
            brightness = float(gray.mean())
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # Relax threshold for dark or blurry frames
            if brightness < 60 or blur_score < 100:
                self.mog2.setVarThreshold(10)
            else:
                self.mog2.setVarThreshold(self.var_threshold)

        raw_mask = self.mog2.apply(frame)
        mask = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections

    def reset(self):
        self._make_mog2()
