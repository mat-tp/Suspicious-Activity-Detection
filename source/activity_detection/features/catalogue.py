"""Looks up what a feature column is: its family, its level and a one-line description.

The column names in ``video_features.csv`` and in the importance files are short, so this
module is the place that says what each one measures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# This is how a track-level value is combined over all tracks of a video (see aggregation.py).
AGGREGATIONS = {
    "mean": "average over the tracks of the video",
    "max": "largest value over the tracks of the video",
    "min": "smallest value over the tracks of the video",
    "std": "spread over the tracks of the video",
    "median": "median over the tracks of the video",
}

# This maps every per-track descriptor to its family and what it measures.
TRACK_DESCRIPTORS: Dict[str, Tuple[str, str]] = {
    "track_duration_sec": ("trajectory", "time the object stays tracked, in seconds"),
    "path_speed_px_s": ("trajectory", "path length divided by duration, in pixels per second"),
    "net_displacement_speed_px_s": ("trajectory", "start-to-end distance divided by duration"),
    "straightness": ("trajectory", "start-to-end distance divided by path length (1 = straight line)"),
    "mean_speed_px_s": ("trajectory", "mean frame-to-frame speed"),
    "max_speed_px_s": ("trajectory", "highest frame-to-frame speed"),
    "std_speed_px_s": ("trajectory", "standard deviation of the speed"),
    "skew_speed": ("trajectory", "skewness of the speed distribution"),
    "kurt_speed": ("trajectory", "kurtosis of the speed distribution"),
    "mean_direction_deg": ("trajectory", "mean heading of the movement, in degrees"),
    "direction_std_deg": ("trajectory", "spread of the heading, in degrees"),
    "bbox_area": ("trajectory", "mean bounding-box area, in pixels"),
    "mean_flow_magnitude": ("trajectory", "mean optical-flow magnitude inside the box"),
    "mean_flow_angle_deg": ("trajectory", "mean optical-flow direction inside the box"),
    "circular_mean_rad": ("circular", "mean heading as a circular statistic, in radians"),
    "circular_variance": ("circular", "circular variance of the heading (0 = one direction)"),
    "direction_concentration": ("circular", "length of the mean heading vector (1 = always the same direction)"),
    "angular_dispersion": ("circular", "spread of the step angles around the circular mean"),
    "mean_accel_px_s2": ("kinematics", "mean acceleration"),
    "std_accel_px_s2": ("kinematics", "standard deviation of the acceleration"),
    "max_accel_px_s2": ("kinematics", "highest acceleration"),
    "skew_accel": ("kinematics", "skewness of the acceleration"),
    "mean_jerk_px_s3": ("kinematics", "mean jerk (change of acceleration)"),
    "std_jerk_px_s3": ("kinematics", "standard deviation of the jerk"),
    "max_jerk_px_s3": ("kinematics", "highest jerk"),
    "skew_jerk": ("kinematics", "skewness of the jerk"),
    "mean_curvature": ("shape", "mean curvature of the path"),
    "std_curvature": ("shape", "standard deviation of the curvature"),
    "turning_rate_rad_s": ("shape", "how fast the heading changes, in radians per second"),
    "convex_hull_area": ("shape", "area of the convex hull of the path"),
    "compactness": ("shape", "4 pi times hull area over hull perimeter squared (1 = circle)"),
    "path_efficiency": ("shape", "straight-line distance over travelled distance"),
    "dwt_energy_approx": ("wavelet", "wavelet energy of the slow (approximation) part of the path"),
    "dwt_energy_detail1": ("wavelet", "wavelet energy of the finest detail band"),
    "dwt_energy_detail2": ("wavelet", "wavelet energy of the middle detail band"),
    "dwt_energy_detail3": ("wavelet", "wavelet energy of the coarsest detail band"),
    "dwt_energy_total": ("wavelet", "total wavelet energy of the path"),
    "dwt_entropy": ("wavelet", "entropy of the wavelet energy distribution"),
    "mean_pairwise_distance": ("interaction", "mean distance to the other tracks"),
    "min_pairwise_distance": ("interaction", "closest distance to another track"),
    "std_pairwise_distance": ("interaction", "spread of the distance to the other tracks"),
    "mean_relative_speed_px_s": ("interaction", "mean speed relative to the other tracks"),
    "max_relative_speed_px_s": ("interaction", "highest speed relative to another track"),
    "proximity_duration_ratio": ("interaction", "share of frames in which another track is closer than the proximity threshold"),
    "proximity_event_rate_s": ("interaction", "number of times another track comes closer than the threshold, per second"),
}

# This lists the video-level features that are not aggregated track descriptors.
VIDEO_FEATURES: Dict[str, Tuple[str, str]] = {
    "n_valid_tracks": ("count", "number of tracks long enough to describe"),
    "indicator_tracks_present": ("count", "1 if the video has at least one usable track"),
    "meta_n_traj_tracks": ("count", "number of tracks that kept a trajectory"),
    "meta_num_frames": ("meta", "frame count of the file"),
    "meta_duration_sec": ("meta", "length of the file in seconds"),
    "meta_fps": ("meta", "frame rate of the file"),
    "temporal_n_tracks": ("temporal", "number of tracks in the video"),
    "temporal_n_long_tracks": ("temporal", "number of tracks above the length threshold"),
    "temporal_spread": ("temporal", "standard deviation of the track lengths"),
    "temporal_spatial_spread": ("temporal", "spread of all track positions in the frame"),
    "temporal_overlap_count": ("temporal", "number of pairs of tracks that both have at least 15 points (an estimate of overlap)"),
    "temporal_tracks_per_frame": ("temporal", "total number of track points divided by the number of frames"),
    "temporal_mean_track_len": ("temporal", "mean track length in frames"),
    "temporal_max_track_len": ("temporal", "longest track in frames"),
    "flow_mean_mag": ("flow", "mean optical-flow magnitude over the whole frame"),
    "flow_std_mag": ("flow", "spread of the whole-frame flow magnitude"),
    "flow_max_mag": ("flow", "highest whole-frame flow magnitude"),
    "flow_prop_moving": ("flow", "share of pixels that move more than one pixel between frame pairs"),
    "flow_dir_entropy": ("flow", "entropy of the eight-bin flow direction histogram"),
    "flow_jerk": ("flow", "mean absolute change of the flow magnitude between frame pairs"),
    "app_grad_energy_mean": ("appearance", "mean gradient energy of the frames"),
    "app_grad_energy_std": ("appearance", "spread of the gradient energy"),
    "diag_brightness_mean": ("image quality", "mean brightness of the frames"),
    "diag_brightness_var": ("image quality", "variance of the brightness over the frames"),
    "diag_dynamic_range": ("image quality", "difference between the brightest and the darkest frame"),
    "diag_low_contrast_ratio": ("image quality", "share of frames with a low sharpness value (Laplacian variance below 50)"),
}

# These are the families whose column names end in a number, with the text for that number.
# Each row is: regular expression, family, description template.
_INDEXED = [
    (r"flow_hof_(mean|std)_(\d+)", "flow", "{0} of the histogram-of-flow bin {1} (direction bin)"),
    (r"app_lbp_(mean|std)_(\d+)", "appearance", "{0} of local-binary-pattern bin {1}"),
    (r"app_hog_(mean|std)_(\d+)", "appearance", "{0} of gradient-orientation bin {1}"),
    (r"diag_(laplacian|tenengrad|brenner)_(mean|std|min|max)",
     "image quality", "{1} of the {0} sharpness measure over the frames"),
    (r"dense_hog_bin_(\d+)", "dense trajectories", "HOG histogram bin {0}"),
    (r"dense_hof_bin_(\d+)", "dense trajectories", "HOF histogram bin {0}"),
    (r"dense_mbh(x|y)_bin_(\d+)", "dense trajectories", "MBH{0} histogram bin {1}"),
    (r"dense_(.+)", "dense trajectories", "dense-trajectory statistic: {0}"),
]


@dataclass(frozen=True)
class FeatureInfo:
    """What one feature column is."""

    name: str
    family: str
    description: str


def describe(name: str) -> Optional[FeatureInfo]:
    """The :class:`FeatureInfo` of a column name, or ``None`` if the name is not known."""
    if name in VIDEO_FEATURES:
        family, text = VIDEO_FEATURES[name]
        return FeatureInfo(name, family, text)
    agg, _, descriptor = name.partition("_")
    if agg in AGGREGATIONS and descriptor in TRACK_DESCRIPTORS:
        family, text = TRACK_DESCRIPTORS[descriptor]
        return FeatureInfo(name, family, f"{text} ({AGGREGATIONS[agg]})")
    for pattern, family, template in _INDEXED:
        match = re.fullmatch(pattern, name)
        if match:
            return FeatureInfo(name, family, template.format(*match.groups()))
    return None


def unknown_features(names: Iterable[str]) -> List[str]:
    """The names in *names* that the catalogue cannot describe."""
    return [name for name in names if describe(name) is None]
