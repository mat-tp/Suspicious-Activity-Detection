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
    def __init__(self, gt_dir=None, frame_resize=(640, 480), fallback_to_mog2=False):
        self.gt_dir = (Path(gt_dir) if gt_dir else DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT")
        self.frame_resize = tuple(frame_resize)
        self.fallback_to_mog2 = fallback_to_mog2
        self._loaded_bboxes: List[Optional[BBox]] = []
        self._frame_idx = 0
        self.current_video = ""

    def load_for_video(self, video_name: str, orig_w: int, orig_h: int) -> bool:
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
                matches = list(self.gt_dir.glob(pattern))
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
        if self._frame_idx < len(self._loaded_bboxes):
            bbox = self._loaded_bboxes[self._frame_idx]
            self._frame_idx += 1
            return [bbox] if bbox is not None else []
        self._frame_idx += 1
        return []

    def reset(self):
        self._frame_idx = 0
