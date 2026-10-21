"""Shared base for detectors built on a per-pixel background model.

A subclass only supplies the background model (segment). Cleaning the mask and
extracting boxes is identical for every detector here, and matches MOG2, so a
comparison between detectors measures the background model and nothing else.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..interfaces import BBox, Detector
from .utils import _to_gray, boxes_from_mask, clean_binary_mask


class BackgroundModelDetector(Detector):
    """Base class of the background models: a subclass implements ``segment``.

    This class cleans the mask and extracts the boxes in the same way for every model.

    Args:
        morph_close_iter, morph_open_iter, min_contour_area: see ``MOG2Detector``.
    """

    def __init__(self, morph_close_iter: int = 2, morph_open_iter: int = 1,
                 min_contour_area: int = 500):
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.min_contour_area = min_contour_area

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """Return a boolean foreground mask for this frame and update the model."""
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Boxes of the foreground blobs of *frame* (converted to grey first)."""
        foreground = self.segment(_to_gray(frame))
        mask = clean_binary_mask(foreground, self.morph_close_iter, self.morph_open_iter)
        return boxes_from_mask(mask, self.min_contour_area)
