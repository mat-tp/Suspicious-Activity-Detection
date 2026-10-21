"""Detector that replays the ground-truth boxes (an upper bound for tracking and classification)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..detection.utils import make_bbox
from ..interfaces import BBox, Detector
from ..paths import DATA_DIR
from ..registry import register


@register("detector", "ground_truth")
class GroundTruthDetector(Detector):
    """Returns the annotated box of the current frame instead of detecting anything.

    AD-SVD annotates one region per sequence (every 5th frame, interpolated in between). The boxes come from
    a text file with one ``x y w h`` line per frame and are rescaled to the processed frame size, so
    ``frame_resize`` must equal the preprocessor's. The n-th ``detect`` call returns the n-th line, so use
    ``frame_sampling=1``.

    Args:
        gt_dir: folder with the annotation files.
        frame_resize: (width, height) of the processed frames.
        fallback_to_mog2: for a video without annotation the runner uses MOG2 for that video only.
    """

    def __init__(self, gt_dir=None, frame_resize=(640, 480), fallback_to_mog2=False):
        self.gt_dir = (Path(gt_dir) if gt_dir else DATA_DIR / "surveillanceVideosGT")
        self.frame_resize = tuple(frame_resize)
        self.fallback_to_mog2 = fallback_to_mog2
        # This is the boxes of the current video, one entry per frame (None where there
        # is no annotation).
        self._loaded_bboxes: List[Optional[BBox]] = []
        # This is the number of detect() calls made on the current video.
        self._frame_idx = 0
        # This is the name of the video whose annotation is loaded.
        self.current_video = ""

    def load_for_video(self, video_name: str, orig_w: int, orig_h: int) -> bool:
        """Find and parse the annotation file of *video_name*; returns True if boxes were loaded.

        The file is looked up by several naming patterns (shortest file name wins). *orig_w*,
        *orig_h* are the video's original size, used to rescale the boxes.
        """
        self.current_video = video_name
        stem = Path(video_name).stem

        # Try multiple naming patterns
        patterns = [
            f"{stem}_gt.txt",
            f"{stem}.txt",
            f"{stem}_groundtruth.txt",
            f"{stem}_bbox.txt",
            f"*{stem}*_gt.txt",
            f"*{stem}*",
        ]
        gt_files = []
        if self.gt_dir.exists():
            for pattern in patterns:
                # Shortest file name first: the closest match wins over a longer name that
                # merely contains the stem (e.g. "A1" vs "A10").
                matches = sorted(self.gt_dir.glob(pattern), key=lambda m: (len(m.name), m.name))
                if matches:
                    gt_files = matches
                    break
            if not gt_files:
                gt_files = list(self.gt_dir.rglob(f"*{stem}*_gt.txt"))
            if not gt_files:
                gt_files = list(self.gt_dir.rglob(f"*{stem}*"))

        if not gt_files:
            print(f"  [GT] No GT file found for {video_name} (stem={stem})")
            self._loaded_bboxes = []
            return False

        gt_path = gt_files[0]
        try:
            bboxes = []
            with open(gt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        bboxes.append(None)
                        continue
                    # Split on commas or whitespace
                    parts = re.split(r'[,\s]+', line)
                    if len(parts) >= 4:
                        try:
                            x, y, w, h = map(float, parts[:4])
                            scale_x = self.frame_resize[0] / orig_w
                            scale_y = self.frame_resize[1] / orig_h
                            bboxes.append(make_bbox(
                                int(x * scale_x), int(y * scale_y),
                                int(w * scale_x), int(h * scale_y)
                            ))
                        except ValueError:
                            bboxes.append(None)
                    else:
                        bboxes.append(None)

            self._loaded_bboxes = bboxes
            self._frame_idx = 0
            n_bboxes = len([b for b in bboxes if b is not None])
            print(f"  [GT] Loaded {n_bboxes} bboxes over {len(bboxes)} frames from {gt_path.name}")
            if n_bboxes == 0 and len(bboxes) > 0:
                # Show first line for debugging
                first_line = None
                with open(gt_path) as f2:
                    for line in f2:
                        if line.strip():
                            first_line = line.strip()
                            break
                print(f"  [GT] Debug: first line = '{first_line}' (split into {len(re.split(r'[,\s]+', first_line))} parts)")
            return True

        except Exception as e:
            print(f"  [GT] Failed to parse {gt_path}: {e}")
            self._loaded_bboxes = []
            return False

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """The ground-truth box of the current frame as a list, or ``[]`` if there is none. *frame* is ignored.
        """
        if self._frame_idx < len(self._loaded_bboxes):
            bbox = self._loaded_bboxes[self._frame_idx]
            self._frame_idx += 1
            return [bbox] if bbox is not None else []
        self._frame_idx += 1
        return []

    def reset(self):
        """Rewind to the first annotated frame (keeps the loaded annotation)."""
        self._frame_idx = 0
