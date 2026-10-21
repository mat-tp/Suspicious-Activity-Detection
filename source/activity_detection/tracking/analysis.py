"""Descriptive statistics of the tracks of one video (used for reporting, not for classification)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..interfaces import Track


class TrackerAnalyzer:
    """Summaries of how well a tracker did: track-length distribution, coverage and continuity."""

    @staticmethod
    def compute_tracker_metrics(tracks: List[Track], total_frames: int) -> Dict[str, Any]:
        """Track-length statistics, coverage, tracks per frame and the number of tracks of at least 15/30/45/60/90 points.

        ``tracks_with_gaps`` counts tracks unmatched at the end of the video, so it understates mid-video gaps.
        Returns ``{"error": ...}`` when there are no tracks.
        """
        if not tracks:
            return {"error": "No tracks available"}
        track_lengths = [len(t["trajectory"]) for t in tracks]
        metrics = {
            "n_tracks": len(tracks),
            "mean_track_length": float(np.mean(track_lengths)),
            "median_track_length": float(np.median(track_lengths)),
            "std_track_length": float(np.std(track_lengths)),
            "max_track_length": int(np.max(track_lengths)),
            "min_track_length": int(np.min(track_lengths)),
            "track_coverage": float(np.sum(track_lengths) / (len(tracks) * total_frames)) if total_frames > 0 else 0.0,
            "tracks_per_frame": float(len(tracks) / total_frames) if total_frames > 0 else 0.0,
        }
        thresholds = [15, 30, 45, 60, 90]
        for thresh in thresholds:
            long_tracks = sum(1 for tlen in track_lengths if tlen >= thresh)
            metrics[f"tracks_ge_{thresh}_frames"] = long_tracks
            metrics[f"tracks_ge_{thresh}_pct"] = (long_tracks / len(tracks) * 100 if tracks else 0.0)
        tracks_with_gaps = sum(1 for t in tracks if t.get("disappeared", 0) > 0)
        metrics["tracks_with_gaps"] = tracks_with_gaps
        metrics["track_continuity"] = (1.0 - tracks_with_gaps / len(tracks) if tracks else 0.0)
        tracker_types = set(t.get("tracker_type", "unknown") for t in tracks)
        metrics["tracker_types"] = list(tracker_types)
        return metrics

    @staticmethod
    def justify_min_trajectory_length(track_lengths: List[int]) -> Dict[str, Any]:
        """Share of tracks kept at each minimum length from 5 to 100; recommends the first of 30/45/60 that keeps 80%.
        """
        if not track_lengths:
            return {"error": "No track lengths available"}
        justification = {}
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            justification[f"percentile_{p}"] = float(np.percentile(track_lengths, p))
        total_tracks = len(track_lengths)
        thresholds = range(5, 101, 5)
        retention_data = {}
        for thresh in thresholds:
            retained = sum(1 for length in track_lengths if length >= thresh)
            retention_data[str(thresh)] = {
                "retained_tracks": retained,
                "retention_rate": (retained / total_tracks * 100 if total_tracks > 0 else 0.0),
                "mean_length_retained": (float(np.mean([length for length in track_lengths if length >= thresh])) if retained > 0 else 0.0),
            }
        justification["retention_by_threshold"] = retention_data
        for thresh in [30, 45, 60]:
            if retention_data[str(thresh)]["retention_rate"] >= 80:
                justification["recommended_threshold"] = thresh
                justification["recommendation_reason"] = (
                    f"Threshold of {thresh} frames retains "
                    f"{retention_data[str(thresh)]['retention_rate']:.1f}% "
                    f"of tracks while ensuring sufficient trajectory length "
                    f"for robust feature extraction"
                )
                break
        return justification
