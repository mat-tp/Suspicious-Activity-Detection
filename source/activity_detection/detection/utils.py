"""Helpers shared by all detectors: box construction, grey conversion and mask cleaning."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

from ..interfaces import BBox

# This is the 3x3 structuring element (all eight neighbours) that connected-component labelling
# uses to join pixels.
STRUCT = generate_binary_structure(2, 2)


def make_bbox(x: int, y: int, w: int, h: int) -> BBox:
    """Build the box dict used throughout the pipeline from top-left corner and size.

    Keys: ``x, y, w, h``, centre ``cx, cy`` (integer, centre of the box), ``centroid`` = ``(cx, cy)``
    and ``area`` = ``w * h`` (box area, not contour area).
    """
    cx = int(x + w // 2)
    cy = int(y + h // 2)
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cx": cx, "cy": cy, "centroid": (cx, cy), "area": int(w * h)}


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert *frame* to a single-channel grayscale image.

    The pipeline calls detector.detect(pre_result.gray), which is already a
    1-channel uint8 array.  If someone passes a 3- or 4-channel BGR frame
    instead, this function handles both cases so detectors never crash with
    "Bad number of channels".
    """
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return frame if frame.ndim == 2 else frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def clean_binary_mask(binary: np.ndarray, close_iter: int, open_iter: int) -> np.ndarray:
    """Close small holes, then remove small specks. Returns a 0/255 uint8 mask."""
    closed = binary_closing(binary, structure=STRUCT, iterations=close_iter)
    opened = binary_opening(closed, structure=STRUCT, iterations=open_iter)
    return opened.astype(np.uint8) * 255


def boxes_from_mask(mask: np.ndarray, min_area: float) -> list:
    """Bounding boxes of the blobs in a 0/255 mask that are at least min_area pixels."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(make_bbox(x, y, w, h))
    return boxes
