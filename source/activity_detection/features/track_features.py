"""Per-track motion descriptors: speed, direction, acceleration, jerk and path shape.

Each extractor gets one track and returns a dict of numbers, or ``None`` if the track is too short.
Speeds are in pixels per second, so videos with different frame rates can be compared.

A track can go unmatched for a few frames, so two points may be more than one frame apart. By default
every step counts as one frame (``gap_aware=False``), which overstates the speed after a gap and, because
gaps are more common in blurred video, can make a motion feature partly encode the imaging condition.
With ``gap_aware=True`` the real time between the points is used.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from scipy.stats import kurtosis, skew

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..registry import register

# A sample whose spread is below this has no meaningful skewness or kurtosis (scipy returns NaN).
MIN_SPREAD_FOR_SHAPE_STATISTICS = 1e-9


def safe_skew(values) -> float:
    """Skewness, or 0.0 when there are fewer than 3 values or they are (nearly) constant."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 2 or np.ptp(values) < MIN_SPREAD_FOR_SHAPE_STATISTICS:
        return 0.0
    return float(np.nan_to_num(skew(values)))


def safe_kurtosis(values) -> float:
    """Excess kurtosis, or 0.0 when there are fewer than 3 values or they are (nearly) constant."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 2 or np.ptp(values) < MIN_SPREAD_FOR_SHAPE_STATISTICS:
        return 0.0
    return float(np.nan_to_num(kurtosis(values)))


def step_gaps(track: Track, n_points: int, gap_aware: bool) -> np.ndarray:
    """Number of frames between consecutive trajectory points (length ``n_points - 1``).

    All ones unless *gap_aware* is true AND the track records one frame index per point
    (trackers that keep no ``frames`` list, or whose lists disagree in length, fall back to ones).
    """
    ones = np.ones(max(n_points - 1, 0))
    if not gap_aware:
        return ones
    frames = track.get("frames")
    if frames is None or len(frames) != n_points:
        return ones
    gaps = np.diff(np.asarray(frames, dtype=float))
    return np.maximum(gaps, 1.0)


@register("feature", "trajectory")
class TrajectoryFeatures(FeatureExtractor):
    """Speed, heading, size and flow descriptors of a track, in physical units.

    Speeds are multiplied by the frame rate (``context["fps"]``, default 10) to give pixels per second, and
    the duration is the track's own length in seconds. The headings are plain means in [0, 360) and so wrap
    around; ``CircularMotionFeatures`` has the circular versions.

    Args:
        min_trajectory_len: kept for symmetry with the other extractors; the runner does the length filtering.
        gap_aware: use the real time between points (see the module docstring).
    """

    def __init__(self, min_trajectory_len=30, gap_aware=False):
        self.min_trajectory_len = min_trajectory_len
        self.gap_aware = gap_aware

    def feature_names(self) -> List[str]:
        """Names of the 14 descriptors returned by :meth:`extract_track`."""
        return [
            "track_duration_sec",           # track's own duration (s)
            "path_speed_px_s",              # path_length / duration (px/s)
            "net_displacement_speed_px_s",  # net_disp / duration  (px/s)
            "straightness",                 # net_disp / path_len  ∈ [0,1]
            "mean_speed_px_s",              # mean instantaneous speed (px/s)
            "max_speed_px_s",               # max  instantaneous speed (px/s)
            "std_speed_px_s",               # std  of speed (px/s)
            "skew_speed",                   # dimensionless skewness
            "kurt_speed",                   # dimensionless kurtosis
            "mean_direction_deg",           # mean heading (°)
            "direction_std_deg",            # std of heading (°)
            "bbox_area",                    # spatial extent (px²)
            "mean_flow_magnitude",          # video-level LK flow magnitude
            "mean_flow_angle_deg",          # video-level LK flow angle (°)
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Speed, straightness, heading and extent of one track; ``None`` below 2 points.

        ``context`` supplies ``fps`` and the video-level Lucas-Kanade flow summary, which is
        copied unchanged into every track of the video.
        """
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None

        fps = float(context.get("fps", 10.0))
        gaps = step_gaps(track, len(traj), self.gap_aware)
        n_frames_tracked = len(traj)
        # Legacy: one frame per point. Gap-aware: first to last observed frame (+1 frame).
        duration_sec = (gaps.sum() + 1) / fps if self.gap_aware else n_frames_tracked / fps

        deltas      = np.diff(traj, axis=0)
        speeds_pf   = np.linalg.norm(deltas, axis=1) / gaps   # px / frame
        speeds_ps   = speeds_pf * fps                   # px / s
        angles_deg  = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360

        total_dist_px = float(np.linalg.norm(deltas, axis=1).sum())
        net_disp_px   = float(euclidean(traj[0], traj[-1]))
        straightness  = net_disp_px / total_dist_px if total_dist_px > 0 else 0.0

        bbox_wh   = traj.max(axis=0) - traj.min(axis=0)
        bbox_area = float(bbox_wh[0] * bbox_wh[1])

        return {
            "track_duration_sec":           duration_sec,
            "path_speed_px_s":              total_dist_px / duration_sec if duration_sec > 0 else 0.0,
            "net_displacement_speed_px_s":  net_disp_px   / duration_sec if duration_sec > 0 else 0.0,
            "straightness":                 straightness,
            "mean_speed_px_s":              float(speeds_ps.mean()),
            "max_speed_px_s":               float(speeds_ps.max()),
            "std_speed_px_s":               float(speeds_ps.std()),
            "skew_speed":                   safe_skew(speeds_ps),
            "kurt_speed":                   safe_kurtosis(speeds_ps),
            "mean_direction_deg":           float(angles_deg.mean()),
            "direction_std_deg":            float(angles_deg.std()),
            "bbox_area":                    bbox_area,
            "mean_flow_magnitude":          context.get("mean_flow_magnitude", 0.0),
            "mean_flow_angle_deg":          context.get("mean_flow_angle_deg", 0.0),
        }


@register("feature", "circular_motion")
class CircularMotionFeatures(FeatureExtractor):
    """Circular statistics of the heading angles of a track (no wrap-around artefacts).

    Returns the circular mean, circular variance ``1 - R``, the mean resultant length ``R``
    (direction concentration; 1 = perfectly straight) and the angular dispersion.
    """

    def feature_names(self) -> List[str]:
        """Names of the four circular descriptors."""
        return ["circular_mean_rad", "circular_variance", "direction_concentration", "angular_dispersion"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Circular heading statistics of one track; ``None`` below 2 points."""
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None
        deltas = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        circ_mean = np.arctan2(mean_sin, mean_cos)
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        circ_var = 1.0 - R
        ang_diff = np.arctan2(np.sin(angles_rad - circ_mean), np.cos(angles_rad - circ_mean))
        return {
            "circular_mean_rad": float(circ_mean),
            "circular_variance": float(circ_var),
            "direction_concentration": float(R),
            "angular_dispersion": float(np.std(ang_diff)),
        }


@register("feature", "kinematics")
class KinematicsFeatures(FeatureExtractor):
    """Acceleration and jerk in pixels per second squared and cubed (frame rate from ``context["fps"]``).

    Args:
        gap_aware: divide by the real time between points (see the module docstring).
    """

    def __init__(self, gap_aware: bool = False):
        self.gap_aware = gap_aware

    def feature_names(self) -> List[str]:
        """Names of the eight acceleration and jerk descriptors."""
        return [
            "mean_accel_px_s2", "std_accel_px_s2", "max_accel_px_s2", "skew_accel",
            "mean_jerk_px_s3",  "std_jerk_px_s3",  "max_jerk_px_s3",  "skew_jerk",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Acceleration and jerk statistics of one track; ``None`` below 4 points."""
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None
        fps = float(context.get("fps", 10.0))

        gaps       = step_gaps(track, len(traj), self.gap_aware)
        deltas     = np.diff(traj, axis=0)
        speeds_pf  = np.linalg.norm(deltas, axis=1) / gaps   # px/frame
        # Time (in frames) between successive speed / acceleration samples.
        dt_speed   = (gaps[:-1] + gaps[1:]) / 2.0
        accel_ps2  = np.diff(speeds_pf) / dt_speed * fps       # px/s²
        dt_accel   = (dt_speed[:-1] + dt_speed[1:]) / 2.0 if len(dt_speed) > 1 else dt_speed[:0]
        jerk_ps3   = np.diff(accel_ps2) / dt_accel * fps if len(dt_accel) else np.array([])   # px/s³

        if len(accel_ps2) < 2 or len(jerk_ps3) < 2:
            return None
        return {
            "mean_accel_px_s2": float(np.mean(accel_ps2)),
            "std_accel_px_s2":  float(np.std(accel_ps2)),
            "max_accel_px_s2":  float(np.max(np.abs(accel_ps2))),
            "skew_accel":       safe_skew(accel_ps2),
            "mean_jerk_px_s3":  float(np.mean(jerk_ps3)),
            "std_jerk_px_s3":   float(np.std(jerk_ps3)),
            "max_jerk_px_s3":   float(np.max(np.abs(jerk_ps3))),
            "skew_jerk":        safe_skew(jerk_ps3),
        }


@register("feature", "shape")
class ShapeFeatures(FeatureExtractor):
    """Curvature, turning rate, hull area, compactness and path efficiency of a track.

    Args:
        gap_aware: use the real track duration for ``turning_rate_rad_s`` (see the module docstring).
    """

    def __init__(self, gap_aware: bool = False):
        self.gap_aware = gap_aware

    def feature_names(self) -> List[str]:
        """Names of the six shape descriptors."""
        return [
            "mean_curvature",      # mean |Δangle| per step (rad)
            "std_curvature",       # std  |Δangle| per step (rad)
            "turning_rate_rad_s",  # total angle turned / duration (rad/s)
            "convex_hull_area",    # spatial extent (px²)
            "compactness",         # 4π·area/perimeter²
            "path_efficiency",     # net_disp / path_len ∈ [0,1]
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Curvature, turning rate, hull area, compactness and path efficiency; ``None`` below 4 points."""
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        fps          = float(context.get("fps", 10.0))
        gaps         = step_gaps(track, len(traj), self.gap_aware)
        duration_sec = ((gaps.sum() + 1) if self.gap_aware else len(traj)) / fps

        deltas     = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        curvature  = np.abs(np.diff(angles_rad))
        curvature  = np.minimum(curvature, 2 * np.pi - curvature)

        total_angle   = float(curvature.sum())
        turning_rate  = total_angle / duration_sec if duration_sec > 0 else 0.0

        hull_area   = 0.0
        compactness = 0.0
        try:
            if len(traj) >= 3:
                hull = ConvexHull(traj)
                hull_area  = float(hull.volume)
                hull_perim = float(hull.area)
                if hull_perim > 0:
                    compactness = 4 * np.pi * hull_area / (hull_perim ** 2)
        except Exception:
            pass

        total_dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net_disp   = float(euclidean(traj[0], traj[-1]))
        path_eff   = net_disp / total_dist if total_dist > 0 else 0.0

        return {
            "mean_curvature":     float(np.mean(curvature)) if len(curvature) > 0 else 0.0,
            "std_curvature":      float(np.std(curvature))  if len(curvature) > 0 else 0.0,
            "turning_rate_rad_s": turning_rate,
            "convex_hull_area":   hull_area,
            "compactness":        compactness,
            "path_efficiency":    path_eff,
        }
