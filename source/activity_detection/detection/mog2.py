"""MOG2 background-subtraction detector (the baseline of the study)."""

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
    """MOG2 background subtraction followed by mask cleaning and box extraction.

    Per frame: OpenCV MOG2, keep sure-foreground pixels (value >= 200), close then open the mask, and make
    one box per contour of at least ``min_contour_area`` pixels.

    Args:
        history: number of frames that define the background model.
        var_threshold: squared Mahalanobis distance above which a pixel is foreground. 16 (4 sigma) is the
            value of Zivkovic and van der Heijden (2006) and OpenCV's default; justify any other value with
            ``scripts/mog2_threshold_sweep.py``.
        detect_shadows: use OpenCV's shadow model (shadow pixels are 127 in the raw mask).
        shadow_value: grey value that OpenCV uses for shadows.
        shadow_as_foreground_weight: 0 = shadows are never foreground, 1 = always, in between = only
            shadow pixels that touch sure foreground.
        morph_close_iter, morph_open_iter: closing (fills holes) and opening (removes specks) iterations.
        min_contour_area: smaller blobs are dropped (pixels of the preprocessed frame).
        adaptive: crude rule that lowers the threshold to 10 on dark (mean < 60) or blurry (Laplacian
            variance < 100) frames; ``AdaptiveMOG2Detector`` is the better version.
    """

    def __init__(self, history=20, var_threshold=16, detect_shadows=True,
                 shadow_value=127, shadow_as_foreground_weight=0.0,
                 morph_close_iter=2, morph_open_iter=1, min_contour_area=500,
                 adaptive=False):
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
        """Turn the raw MOG2 mask (0 background, 127 shadow, 255 foreground) into a clean 0/255 mask."""
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
        """Boxes of the moving blobs in *frame*; this also updates the background model."""
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
