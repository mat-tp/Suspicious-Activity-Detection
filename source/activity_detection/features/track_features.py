from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from scipy.stats import kurtosis, skew

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..registry import register


@register("feature", "trajectory")
class TrajectoryFeatures(FeatureExtractor):
    """
    Trajectory-level motion descriptors, fully normalised to physical units.

    Normalisation (mentor v6.0 — applied identically to ALL groups A–I):
      speeds    [px/frame]  × fps   → px/s
      path_len, net_disp             ÷ duration_sec → px/s (rates)
      track_duration               = n_frames / fps (seconds)
      dimensionless ratios (straightness, skewness, kurtosis,
        direction_std_deg) — unchanged.

    fps is read from context["fps"] (injected by PipelineRunner from
    CAP_PROP_FPS; default 10.0 for backward compatibility).
    Per-track duration uses the track's own frame count, NOT the total
    video duration, so a short track inside a long video is correctly
    normalised.  Research: mentor review v6.0; Sargano et al. (2017).
    """

    def __init__(self, min_trajectory_len=30):
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
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
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None

        fps = float(context.get("fps", 10.0))
        n_frames_tracked = len(traj)
        duration_sec = n_frames_tracked / fps      # this track's own duration

        deltas      = np.diff(traj, axis=0)
        speeds_pf   = np.linalg.norm(deltas, axis=1)   # px / frame
        speeds_ps   = speeds_pf * fps                   # px / s
        angles_deg  = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360

        total_dist_px = float(speeds_pf.sum())
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
            "skew_speed":                   float(skew(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "kurt_speed":                   float(kurtosis(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "mean_direction_deg":           float(angles_deg.mean()),
            "direction_std_deg":            float(angles_deg.std()),
            "bbox_area":                    bbox_area,
            "mean_flow_magnitude":          context.get("mean_flow_magnitude", 0.0),
            "mean_flow_angle_deg":          context.get("mean_flow_angle_deg", 0.0),
        }


@register("feature", "circular_motion")
class CircularMotionFeatures(FeatureExtractor):
    def feature_names(self) -> List[str]:
        return ["circular_mean_rad", "circular_variance", "direction_concentration", "angular_dispersion"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
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
    """
    Acceleration and jerk in physical units (mentor v6.0).

      speed     [px/frame]  × fps   = speed  [px/s]
      accel     Δspeed/Δt   × fps   = accel  [px/s²]  (Δt = 1/fps s)
      jerk      Δaccel/Δt   × fps   = jerk   [px/s³]

    Skewness statistics are dimensionless and unchanged.
    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_accel_px_s2", "std_accel_px_s2", "max_accel_px_s2", "skew_accel",
            "mean_jerk_px_s3",  "std_jerk_px_s3",  "max_jerk_px_s3",  "skew_jerk",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None
        fps = float(context.get("fps", 10.0))

        deltas     = np.diff(traj, axis=0)
        speeds_pf  = np.linalg.norm(deltas, axis=1)   # px/frame
        accel_ps2  = np.diff(speeds_pf) * fps          # px/s²  (×fps converts Δ/frame to Δ/s)
        jerk_ps3   = np.diff(accel_ps2) * fps          # px/s³

        if len(accel_ps2) < 2 or len(jerk_ps3) < 2:
            return None
        return {
            "mean_accel_px_s2": float(np.mean(accel_ps2)),
            "std_accel_px_s2":  float(np.std(accel_ps2)),
            "max_accel_px_s2":  float(np.max(np.abs(accel_ps2))),
            "skew_accel":       float(skew(accel_ps2)) if len(accel_ps2) > 2 else 0.0,
            "mean_jerk_px_s3":  float(np.mean(jerk_ps3)),
            "std_jerk_px_s3":   float(np.std(jerk_ps3)),
            "max_jerk_px_s3":   float(np.max(np.abs(jerk_ps3))),
            "skew_jerk":        float(skew(jerk_ps3))  if len(jerk_ps3)  > 2 else 0.0,
        }


@register("feature", "shape")
class ShapeFeatures(FeatureExtractor):
    """
    Trajectory shape descriptors (mentor v6.0 normalisation).

      mean/std curvature: |Δangle| per step (rad) — already dimensionless.
      turning_rate_rad_s: total angle turned ÷ duration_sec → rad/s.
        Captures direction-change rate independently of video length.
      convex_hull_area:   spatial extent in px² — length-independent within
        a video; kept raw.
      compactness:        4π·area/perimeter² ∈ [0,1] — dimensionless.
      path_efficiency:    net_disp / path_len ∈ [0,1] — dimensionless.

    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_curvature",      # mean |Δangle| per step (rad)
            "std_curvature",       # std  |Δangle| per step (rad)
            "turning_rate_rad_s",  # total angle turned / duration (rad/s)
            "convex_hull_area",    # spatial extent (px²)
            "compactness",         # 4π·area/perimeter²
            "path_efficiency",     # net_disp / path_len ∈ [0,1]
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        fps          = float(context.get("fps", 10.0))
        duration_sec = len(traj) / fps

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
