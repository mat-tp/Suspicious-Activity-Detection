"""Regression tests for defects in the pipeline runner; each fails on the old code.

They cover empty frame lists for flow and appearance features, a ground-truth fallback that replaced the
configured detector, a swallowed extractor crash, tracks lost when they were retired mid-video, and the
preprocessors reporting different quality measurements.
"""
import logging

import cv2
import numpy as np
import pytest

from activity_detection.detection.mog2 import MOG2Detector
from activity_detection.features.track_features import KinematicsFeatures, TrajectoryFeatures
from activity_detection.interfaces import FeatureExtractor
from activity_detection.pipeline.runner import PipelineRunner
from activity_detection.preprocessing import (
    DistortionAwarePreprocessor, PassthroughPreprocessor, StandardPreprocessor,
    frame_quality_metrics)
from activity_detection.tracking.iou_hungarian import IoUHungarianTracker


def _write_video(path, n_frames=60, leave_at=None, size=(160, 120)):
    """A bright square that moves right; it disappears at frame ``leave_at`` if given."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = (rng.normal(60, 2, (size[1], size[0], 3))).clip(0, 255).astype(np.uint8)
        if leave_at is None or i < leave_at:
            x = 5 + i
            frame[40:80, x:x + 30] = 230
        w.write(frame)
    w.release()
    return path


def _runner(**kw):
    return PipelineRunner(
        StandardPreprocessor(frame_resize=(160, 120)),
        MOG2Detector(history=10, var_threshold=16, detect_shadows=False, min_contour_area=100),
        IoUHungarianTracker(max_disappeared=3), **kw)


def test_multifamily_flow_and_appearance_features_are_not_all_zero(tmp_path):
    video = _write_video(tmp_path / "move.avi")
    result = _runner(enable_multifamily=True).process_video(video)
    fv = result.video_feature
    assert fv is not None
    assert fv["flow_mean_mag"] > 0 or fv["flow_max_mag"] > 0, "global flow features are still zero"
    assert any(v != 0 for k, v in fv.items() if k.startswith("app_")), "appearance features are zero"


def test_detector_is_restored_after_each_video(tmp_path):
    runner = _runner()
    configured = runner.detector

    class Boom(Exception):
        pass

    def broken(*a, **k):
        runner.detector = object()          # simulate the ground-truth fallback swap
        raise Boom

    runner._process_video = broken
    with pytest.raises(Boom):
        runner.process_video(tmp_path / "x.avi")
    assert runner.detector is configured


def test_failing_extractor_is_logged_and_counted(tmp_path, caplog):
    class Crashing(FeatureExtractor):
        def feature_names(self):
            return ["x"]

        def extract_track(self, track, **context):
            raise RuntimeError("bug in descriptor")

    video = _write_video(tmp_path / "move.avi")
    runner = _runner(feature_extractors=[Crashing()], enable_multifamily=True)
    with caplog.at_level(logging.WARNING):
        result = runner.process_video(video)
    assert result.tracking_stats["extractor_errors"] > 0
    assert "bug in descriptor" in caplog.text


def test_track_that_left_the_scene_still_yields_a_usable_trajectory(tmp_path):
    video = _write_video(tmp_path / "leaves.avi", n_frames=60, leave_at=20)
    result = _runner(min_trajectory_len=5).process_video(video)
    assert result.track_lengths, "the retired track never reached feature extraction"
    assert result.n_usable_tracks >= 1


def test_all_preprocessors_report_the_same_quality_keys():
    frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    keys = {"laplacian_var", "tenengrad", "brenner", "brightness"}
    for pre in (PassthroughPreprocessor(), StandardPreprocessor((160, 120)),
                DistortionAwarePreprocessor(frame_resize=(160, 120))):
        assert keys <= set(pre.process(frame).diagnostics), type(pre).__name__


def test_quality_metrics_separate_sharp_from_blurred():
    sharp = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
    assert frame_quality_metrics(sharp)["laplacian_var"] > frame_quality_metrics(blurred)["laplacian_var"]


def _gapped_track():
    """Moves 10 px per frame, but frames 3-5 were missed (points at frames 0,1,2,6,7,8)."""
    frames = [0, 1, 2, 6, 7, 8]
    xs = [10 * f for f in frames]
    return {"trajectory": [(float(x), 0.0) for x in xs], "frames": frames}


def test_gap_aware_speed_is_not_inflated_by_missed_frames():
    track = _gapped_track()
    legacy = TrajectoryFeatures(gap_aware=False).extract_track(track, fps=10.0)
    aware = TrajectoryFeatures(gap_aware=True).extract_track(track, fps=10.0)
    assert aware["mean_speed_px_s"] == pytest.approx(100.0)       # 10 px/frame * 10 fps
    assert aware["max_speed_px_s"] == pytest.approx(100.0)
    assert legacy["max_speed_px_s"] > 300                         # the jump counted as one frame


def test_gap_aware_acceleration_of_constant_speed_is_zero():
    out = KinematicsFeatures(gap_aware=True).extract_track(_gapped_track(), fps=10.0)
    assert out["max_accel_px_s2"] == pytest.approx(0.0, abs=1e-6)
    legacy = KinematicsFeatures(gap_aware=False).extract_track(_gapped_track(), fps=10.0)
    assert legacy["max_accel_px_s2"] > 100
