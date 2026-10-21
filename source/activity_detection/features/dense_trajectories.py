"""Dense-trajectory style descriptors (HOG / HOF / MBH) around interest points of the video."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..registry import register


@register("feature", "dense_trajectories")
class DenseTrajectoryFeatures(FeatureExtractor):
    """Simplified dense-trajectory descriptors.

    Each track gives simple trajectory statistics. The HOG, HOF, MBH, keypoint, flow and edge values are
    computed once per video from the middle pair of frames (``context['prev_gray']`` and
    ``context['curr_gray']``), so they are the same for every track of a video.

    Args:
        min_trajectory_len: shorter tracks return ``None``.
        step_size: grid spacing of the keypoints (px).
        block_size: side of the patch around a keypoint.
        feature_types: any of ``trajectory``, ``hog``, ``hof``, ``mbh``.
    """

    def __init__(self, min_trajectory_len=15, step_size=5, block_size=32, feature_types=None):
        self.min_trajectory_len = min_trajectory_len
        self.step_size = step_size
        self.block_size = block_size
        self.feature_types = feature_types or ["trajectory", "hog", "hof", "mbh"]
        if not hasattr(cv2, "HOGDescriptor"):
            raise ImportError("Dense-trajectory features need cv2.HOGDescriptor, which OpenCV 5 "
                              "removed. Install OpenCV 4: pip install \"opencv-python>=4.8,<5\".")
        # This is the OpenCV HOG descriptor that computes the histograms of a patch.
        self.hog_descriptor = cv2.HOGDescriptor(_winSize=(block_size, block_size), _blockSize=(16, 16),
                                                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
        # This is the FAST corner detector that finds interest points.
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=20)

    def feature_names(self) -> List[str]:
        """Names of all descriptors enabled by ``feature_types``."""
        names = []
        if "trajectory" in self.feature_types:
            names.extend(["dense_traj_length", "dense_traj_displacement", "dense_traj_mean_speed",
                          "dense_traj_std_speed", "dense_traj_direction_mean", "dense_traj_direction_std"])
        if "hog" in self.feature_types:
            names.extend([f"dense_hog_bin_{i}" for i in range(9)])
        if "hof" in self.feature_types:
            names.extend([f"dense_hof_bin_{i}" for i in range(9)])
        if "mbh" in self.feature_types:
            names.extend([f"dense_mbhx_bin_{i}" for i in range(9)])
            names.extend([f"dense_mbhy_bin_{i}" for i in range(9)])
        names.extend(["dense_keypoint_density", "dense_flow_magnitude_mean", "dense_flow_magnitude_std",
                      "dense_texture_variance", "dense_edge_density"])
        return names

    def extract_keypoints(self, gray_frame: np.ndarray) -> np.ndarray:
        """Interest points of *gray_frame* (Shi-Tomasi, FAST and textured grid points), one per grid cell."""
        h, w = gray_frame.shape[:2]
        points = []
        shi_tomasi = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01,
                                             minDistance=self.step_size, blockSize=3)
        if shi_tomasi is not None:
            points.extend(shi_tomasi.reshape(-1, 2))
        kp = self.fast_detector.detect(gray_frame, None)
        points.extend([kp_item.pt for kp_item in kp[:50]])
        for y in range(self.step_size, h - self.step_size, self.step_size):
            for x in range(self.step_size, w - self.step_size, self.step_size):
                patch = gray_frame[y-2:y+3, x-2:x+3]
                if np.std(patch) > 10:
                    points.append([float(x), float(y)])
        if points:
            points = np.array(points)
            if len(points) > 1:
                # Grid-based NMS
                cell = self.step_size
                seen: Dict[Tuple[int, int], bool] = {}
                keep = np.zeros(len(points), dtype=bool)
                for i, (px, py) in enumerate(points):
                    key = (int(px) // cell, int(py) // cell)
                    if key not in seen:
                        seen[key] = True
                        keep[i] = True
                points = points[keep]
        return points if len(points) > 0 else np.array([])

    def compute_local_descriptors(self, gray_patch: np.ndarray, flow_patch: np.ndarray) -> Dict[str, float]:
        """HOG, HOF and MBH histograms (9 bins each) of a grey patch and its flow patch; zeros if they cannot be computed.
        """
        features = {}
        if "hog" in self.feature_types and gray_patch.shape[0] >= self.block_size:
            try:
                resized = cv2.resize(gray_patch, (self.block_size, self.block_size))
                hog = self.hog_descriptor.compute(resized)
                if hog is not None:
                    hog = hog.reshape(-1, 9).mean(axis=0)
                    for i in range(min(9, len(hog))):
                        features[f"dense_hog_bin_{i}"] = float(hog[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hog_bin_{i}"] = 0.0
        if "hof" in self.feature_types and flow_patch.size >= 2:
            try:
                mag, ang = cv2.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])
                mask = mag > 1.0
                if mask.sum() > 0:
                    hof, _ = np.histogram(ang[mask], bins=9, range=(0, 2*np.pi))
                    hof = hof.astype(float) / (hof.sum() + 1e-7)
                else:
                    hof = np.zeros(9)
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = float(hof[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = 0.0
        if "mbh" in self.feature_types and flow_patch.size >= 4:
            try:
                grad_x = cv2.Sobel(flow_patch[..., 0], cv2.CV_32F, 1, 0)
                grad_y = cv2.Sobel(flow_patch[..., 1], cv2.CV_32F, 0, 1)
                mbhx, _ = np.histogram(grad_x.flatten(), bins=9, range=(-10, 10))
                mbhy, _ = np.histogram(grad_y.flatten(), bins=9, range=(-10, 10))
                mbhx = mbhx.astype(float) / (mbhx.sum() + 1e-7)
                mbhy = mbhy.astype(float) / (mbhy.sum() + 1e-7)
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = float(mbhx[i])
                    features[f"dense_mbhy_bin_{i}"] = float(mbhy[i])
            except Exception:
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = 0.0
                    features[f"dense_mbhy_bin_{i}"] = 0.0
        return features

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Trajectory statistics of the track plus the video-level descriptors; ``None`` for a short track."""
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        features = {}
        if "trajectory" in self.feature_types:
            deltas = np.diff(traj, axis=0)
            speeds = np.linalg.norm(deltas, axis=1)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            features.update({
                "dense_traj_length": float(len(traj)),
                "dense_traj_displacement": float(np.linalg.norm(traj[-1] - traj[0])),
                "dense_traj_mean_speed": float(np.mean(speeds)),
                "dense_traj_std_speed": float(np.std(speeds)),
                "dense_traj_direction_mean": float(np.mean(angles)),
                "dense_traj_direction_std": float(np.std(angles))
            })
        prev_gray = context.get("prev_gray")
        curr_gray = context.get("curr_gray")
        if prev_gray is not None and curr_gray is not None:
            h, w = curr_gray.shape[:2]
            keypoints = self.extract_keypoints(curr_gray)
            if len(keypoints) > 0:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_mags = np.linalg.norm(flow, axis=2)
                edges = cv2.Canny(curr_gray, 50, 150)
                features.update({
                    "dense_keypoint_density": float(len(keypoints)),
                    "dense_flow_magnitude_mean": float(np.mean(flow_mags)),
                    "dense_flow_magnitude_std": float(np.std(flow_mags)),
                    "dense_texture_variance": float(np.var(curr_gray)),
                    "dense_edge_density": float(np.sum(edges > 0) / edges.size)
                })
                sampled_kps = keypoints[:20]
                for pt in sampled_kps:
                    x, y = int(pt[0]), int(pt[1])
                    x1 = max(0, x - self.block_size//2)
                    y1 = max(0, y - self.block_size//2)
                    x2 = min(w, x + self.block_size//2)
                    y2 = min(h, y + self.block_size//2)
                    if x2 > x1 and y2 > y1:
                        gray_patch = curr_gray[y1:y2, x1:x2]
                        flow_patch = flow[y1:y2, x1:x2]
                        patch_feats = self.compute_local_descriptors(gray_patch, flow_patch)
                        for key, val in patch_feats.items():
                            features[key] = features.get(key, 0.0) + val
                if sampled_kps and len(sampled_kps) > 0:
                    for key in list(features.keys()):
                        if any(key.startswith(prefix) for prefix in ["dense_hog_", "dense_hof_", "dense_mbh"]):
                            features[key] /= len(sampled_kps)
        return features if features else None
