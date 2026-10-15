import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import feature_family_importance as ffi  # noqa: E402
from test_feature_appendix import APPENDIX  # noqa: E402,F401


def test_known_features_map_to_expected_families():
    assert ffi.family("diag_laplacian_mean") == "image quality"
    assert ffi.family("mean_mean_speed_px_s") == "trajectory"
    assert ffi.family("std_mean_accel_px_s2") == "kinematic"
    assert ffi.family("max_dwt_entropy") == "wavelet"
    assert ffi.family("mean_min_pairwise_distance") == "interaction"
    assert ffi.family("mean_turning_rate_rad_s") == "shape"
    assert ffi.family("mean_circular_variance") == "circular"
    assert ffi.family("meta_fps") == "file metadata"
    assert ffi.family("app_lbp_mean_3") == "appearance"


def test_every_documented_descriptor_has_a_family():
    import re
    from activity_detection.features.interaction import InteractionFeatures
    from activity_detection.features.track_features import (
        CircularMotionFeatures, KinematicsFeatures, ShapeFeatures, TrajectoryFeatures)
    from activity_detection.features.wavelet import WaveletFeatures
    for cls in (TrajectoryFeatures, CircularMotionFeatures, KinematicsFeatures, ShapeFeatures,
                WaveletFeatures, InteractionFeatures):
        for name in cls().feature_names():
            assert ffi.family(f"mean_{name}") != "other", name


def test_shares_sum_to_one():
    shares = ffi.by_family({"diag_brenner_mean": 3.0, "mean_mean_speed_px_s": 1.0})
    assert abs(sum(shares.values()) - 1.0) < 1e-9 and shares["image quality"] == 0.75
