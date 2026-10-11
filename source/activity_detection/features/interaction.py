from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial.distance import euclidean

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..registry import register


@register("feature", "interaction")
class InteractionFeatures(FeatureExtractor):
    def __init__(self, proximity_threshold=100.0, min_trajectory_len=30):
        self.proximity_threshold = proximity_threshold
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["mean_pairwise_distance", "min_pairwise_distance", "std_pairwise_distance",
                "mean_relative_speed", "max_relative_speed", "proximity_duration_ratio", "n_proximity_events"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """
        Interaction features with normalised speed and event rate (v6.0).

          relative speed:  px/frame → px/s  (× fps)
          pairwise_dists:  spatial (px) — unchanged (not temporal)
          proximity_duration_ratio: already dimensionless fraction
          proximity_event_rate_s:  n_events / overlap_duration_sec (events/s)

        fps from context["fps"]; default 10.0.
        """
        all_tracks = context.get("all_tracks", [])
        if len(all_tracks) < 2:
            return None
        my_traj = np.array(track["trajectory"], dtype=float)
        if len(my_traj) < self.min_trajectory_len:
            return None

        fps = float(context.get("fps", 10.0))

        pairwise_dists: List[float] = []
        rel_speeds_ps:  List[float] = []
        proximity_frames = 0
        total_frames     = 0
        n_events         = 0

        for other in all_tracks:
            if other["id"] == track["id"]:
                continue
            other_traj = np.array(other["trajectory"], dtype=float)
            min_len = min(len(my_traj), len(other_traj))
            if min_len < 2:
                continue
            in_proximity = False
            for i in range(min_len):
                d = euclidean(my_traj[i], other_traj[i])
                pairwise_dists.append(d)
                total_frames += 1
                if d < self.proximity_threshold:
                    proximity_frames += 1
                    if not in_proximity:
                        n_events += 1
                        in_proximity = True
                else:
                    in_proximity = False
            my_spd_pf    = np.linalg.norm(np.diff(my_traj[:min_len],    axis=0), axis=1)
            other_spd_pf = np.linalg.norm(np.diff(other_traj[:min_len], axis=0), axis=1)
            # Convert differential speed from px/frame to px/s
            rel_speeds_ps.extend(np.abs(my_spd_pf - other_spd_pf) * fps)

        if not pairwise_dists:
            return None

        overlap_sec = total_frames / fps if fps > 0 else 1.0
        return {
            "mean_pairwise_distance":   float(np.mean(pairwise_dists)),
            "min_pairwise_distance":    float(np.min(pairwise_dists)),
            "std_pairwise_distance":    float(np.std(pairwise_dists)),
            "mean_relative_speed_px_s": float(np.mean(rel_speeds_ps)) if rel_speeds_ps else 0.0,
            "max_relative_speed_px_s":  float(np.max(rel_speeds_ps))  if rel_speeds_ps else 0.0,
            "proximity_duration_ratio": proximity_frames / total_frames if total_frames > 0 else 0.0,
            "proximity_event_rate_s":   n_events / overlap_sec if overlap_sec > 0 else 0.0,
        }
