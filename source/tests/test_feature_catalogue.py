"""Every feature the code can produce must be described by the catalogue."""
import cv2
import pytest

import activity_detection.features.dense_trajectories  # noqa: F401  (registers the extractor)
from activity_detection.features.catalogue import describe, unknown_features
from activity_detection.features.dense_trajectories import DenseTrajectoryFeatures
from activity_detection.features.interaction import InteractionFeatures
from activity_detection.features.track_features import (
    CircularMotionFeatures, KinematicsFeatures, ShapeFeatures, TrajectoryFeatures)
from activity_detection.features.video_features import (
    extract_appearance_features, extract_diagnostic_features, extract_global_flow_features,
    extract_temporal_context)
from activity_detection.features.wavelet import WaveletFeatures

TRACK_MODULES = (TrajectoryFeatures, CircularMotionFeatures, KinematicsFeatures, ShapeFeatures,
                 WaveletFeatures, InteractionFeatures)


def test_every_per_track_descriptor_is_described_for_every_aggregation():
    names = [f"{agg}_{descriptor}" for agg in ("mean", "max", "std")
             for cls in TRACK_MODULES for descriptor in cls().feature_names()]
    assert len(names) == 45 * 3
    assert unknown_features(names) == []


def test_every_video_level_feature_is_described():
    tracks = [{"trajectory": [(0, 0)] * 20}, {"trajectory": [(1, 1)] * 20}]
    names = (list(extract_diagnostic_features([])) + list(extract_global_flow_features([]))
             + list(extract_appearance_features([])) + list(extract_temporal_context(tracks, 40))
             + ["n_valid_tracks", "indicator_tracks_present", "meta_n_traj_tracks",
                "meta_num_frames", "meta_duration_sec", "meta_fps"])
    assert unknown_features(names) == []


@pytest.mark.skipif(not hasattr(cv2, "HOGDescriptor"), reason="OpenCV 5 has no HOGDescriptor")
def test_dense_trajectory_features_are_described():
    assert unknown_features(DenseTrajectoryFeatures().feature_names()) == []


def test_a_description_names_the_family_and_the_aggregation():
    info = describe("std_dwt_entropy")
    assert info.family == "wavelet"
    assert "spread over the tracks" in info.description


def test_an_unknown_name_is_reported():
    assert describe("not_a_feature") is None
    assert unknown_features(["mean_straightness", "not_a_feature"]) == ["not_a_feature"]
