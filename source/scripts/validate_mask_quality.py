"""Does the mask-quality score Q_mask track real mask quality?

Q_mask is a hand-designed heuristic, so it needs an external check. AD-SVD ships per-frame ground-truth bounding boxes, so we can check it. For each
frame we

  1. segment with plain MOG2 at the nominal threshold and score the cleaned mask
     with the detector's own Q_mask, and
  2. measure how well that mask agrees with the ground-truth boxes: the F-measure
     between the mask and the union of GT boxes (a box is a coarse stand-in for the
     object silhouette, so absolute values are lower than with pixel masks, but the
     ranking of frames is what matters here).

It then reports the Spearman correlation between Q_mask and that F-measure, overall
and per distortion condition, with a bootstrap interval over videos. A clearly positive
correlation supports Eq. 1; a near-zero one means the score does not measure what the
detector assumes and should be revised or reported as a limitation.

    python source/scripts/validate_mask_quality.py --config group_i_adaptive_mog2 --n 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.config import load_config  # noqa: E402
from activity_detection.detection.adaptive_mog2 import AdaptiveMOG2Detector  # noqa: E402
from activity_detection.detection.ground_truth import GroundTruthDetector  # noqa: E402
from activity_detection.pipeline.builder import PipelineBuilder  # noqa: E402


def boxes_to_mask(shape: Tuple[int, int], boxes: Sequence[dict]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for b in boxes:
        if b is None:
            continue
        x, y, w, h = (int(b[k]) for k in ("x", "y", "w", "h"))
        mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 255
    return mask


def mask_f_measure(mask: np.ndarray, gt: np.ndarray) -> float:
    """F1 between two binary masks; 1.0 if both are empty, 0.0 if only one is."""
    a, b = mask > 0, gt > 0
    if not a.any() and not b.any():
        return 1.0
    tp = float((a & b).sum())
    if tp == 0:
        return 0.0
    precision, recall = tp / a.sum(), tp / b.sum()
    return 2 * precision * recall / (precision + recall)


def video_frame_records(frames: Iterable[np.ndarray], gt_boxes: List[Optional[dict]],
                        detector: AdaptiveMOG2Detector) -> List[Tuple[float, float, bool]]:
    """(Q_mask, F-measure vs GT, GT box present) for every frame after the model warms up."""
    detector.reset()
    records = []
    for i, frame in enumerate(frames):
        mask, contours, _ = detector._segment(frame, detector.var_threshold, -1.0)
        if i <= detector.history // 2:
            continue
        has_gt = i < len(gt_boxes) and gt_boxes[i] is not None
        gt = [gt_boxes[i]] if has_gt else []
        records.append((detector._score_mask(mask, contours),
                        mask_f_measure(mask, boxes_to_mask(mask.shape, gt)), has_gt))
    return records


def video_frame_scores(frames: Iterable[np.ndarray], gt_boxes: List[Optional[dict]],
                       detector: AdaptiveMOG2Detector) -> List[Tuple[float, float]]:
    """(Q_mask, F-measure vs GT) for every frame after the model has warmed up."""
    return [(q, f) for q, f, _ in video_frame_records(frames, gt_boxes, detector)]


def correlation(pairs: List[Tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    q, f = np.array(pairs).T
    if np.ptp(q) == 0 or np.ptp(f) == 0:
        return float("nan")
    return float(spearmanr(q, f).statistic)


def bootstrap_over_videos(per_video: Dict[str, List[Tuple[float, float]]], n_boot: int = 1000,
                          seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    names = list(per_video)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        pairs = [p for i in pick for p in per_video[names[i]]]
        r = correlation(pairs)
        if not np.isnan(r):
            vals.append(r)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="group_i_adaptive_mog2")
    ap.add_argument("--n", type=int, default=10, help="videos per distortion condition")
    ap.add_argument("--out", default="experiments/mask_quality_validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd
    cfg = load_config(args.config)
    df = pd.read_excel(cfg.paths.dataset_excel)
    rng = np.random.default_rng(args.seed)
    pre = PipelineBuilder.build_preprocessor(cfg)
    detector = PipelineBuilder.build_detector(cfg)
    if not isinstance(detector, AdaptiveMOG2Detector):
        raise SystemExit("config must use detection.type: adaptive_mog2 (we reuse its Q_mask scorer)")
    gt = GroundTruthDetector(gt_dir=cfg.paths.gt_dir, frame_resize=tuple(cfg.preprocessing.frame_resize))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # per_cond[condition][video] = list of (q_mask, f_measure, gt_present) per frame
    per_cond: Dict[str, Dict[str, List[Tuple[float, float, bool]]]] = {}
    skipped = {"cannot_open": 0, "no_ground_truth": 0, "no_scored_frames": 0}
    for cond, grp in df.groupby("Distortion"):
        names = grp["Name of Video Series"].tolist()
        for name in rng.permutation(names)[: args.n]:
            path = Path(cfg.paths.video_dir) / name
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                skipped["cannot_open"] += 1
                continue
            w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not gt.load_for_video(name, w0, h0):
                skipped["no_ground_truth"] += 1
                cap.release()
                continue
            frames = []
            while True:
                ok, raw = cap.read()
                if not ok:
                    break
                frames.append(pre.process(raw).bgr)
            cap.release()
            records = video_frame_records(frames, gt._loaded_bboxes, detector)
            if records:
                per_cond.setdefault(str(cond), {})[str(name)] = records
            else:
                skipped["no_scored_frames"] += 1

    with open(out_dir / "frame_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "video", "q_mask", "f_measure_vs_gt_boxes", "gt_present"])
        for cond, vids in per_cond.items():
            for name, recs in vids.items():
                w.writerows((cond, name, q, fm, g) for q, fm, g in recs)

    summary = {"skipped_videos": skipped, "analyses": {}}
    print(f"\nSkipped videos: {skipped}")
    for label, only_gt_frames in (("all scored frames", False), ("frames with a GT box only", True)):
        print(f"\nSpearman(Q_mask, F-measure vs GT) - {label}")
        groups = {"all": {v: r for c in per_cond.values() for v, r in c.items()}, **per_cond}
        summary["analyses"][label] = {}
        for cond, vids in groups.items():
            pairs_by_video = {v: [(q, fm) for q, fm, g in r if g or not only_gt_frames]
                              for v, r in vids.items()}
            pairs_by_video = {v: p for v, p in pairs_by_video.items() if p}
            n_frames = sum(len(p) for p in pairs_by_video.values())
            rho = correlation([p for v in pairs_by_video.values() for p in v])
            lo, hi = bootstrap_over_videos(pairs_by_video)
            print(f"  {cond:<8}: rho = {rho:.3f}  95% CI over videos [{lo:.3f}, {hi:.3f}]  "
                  f"({len(pairs_by_video)} videos, {n_frames} frames)")
            summary["analyses"][label][cond] = {
                "rho": rho, "ci95": [lo, hi], "n_videos": len(pairs_by_video), "n_frames": n_frames}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
