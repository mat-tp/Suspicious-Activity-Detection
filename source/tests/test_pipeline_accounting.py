"""Tests for yield accounting, grouped cross-validation, feature families, frame indices and frame rates."""
import cv2
import numpy as np
import pytest

import activity_detection.evaluation.cross_validation as cv
from activity_detection.config import Config
from activity_detection.detection.utils import make_bbox
from activity_detection.evaluation.cross_validation import cross_validate_model
from activity_detection.evaluation.groups import derive_groups
from activity_detection.experiment.runner import ExperimentRunner
from activity_detection.features.interaction import InteractionFeatures
from activity_detection.features.track_features import safe_kurtosis, safe_skew
from activity_detection.models.imbalance import ImbalanceConfig, build_model
from activity_detection.pipeline.builder import PipelineBuilder
from activity_detection.pipeline.runner import PipelineRunner
from activity_detection.tracking.iou_hungarian import IoUHungarianTracker


def test_derive_groups_from_regex():
    names = ["scene1_Exp_q50.avi", "scene1_Fo_q75.avi", "scene2_Exp_q50.avi"]
    g = derive_groups(names, regex=r"^(scene\d+)_")
    assert g[0] == g[1] != g[2]


def test_derive_groups_none_when_unconfigured():
    assert derive_groups(["a.avi"]) is None


def test_derive_groups_regex_without_match_raises():
    with pytest.raises(ValueError):
        derive_groups(["zzz.avi"], regex=r"^(scene\d+)_")


def test_grouped_cv_never_splits_a_group():
    rng = np.random.default_rng(0)
    n_groups, per_group = 40, 3
    groups = np.repeat(np.arange(n_groups), per_group)
    y = np.repeat(rng.integers(0, 2, n_groups), per_group)
    X = rng.normal(size=(len(y), 4)) + y[:, None]
    model = build_model("random_forest_sklearn", {"n_estimators": 20, "n_jobs": 1},
                        ImbalanceConfig.from_config(None), 0)
    seen = {}

    original = cv._split

    def spy(X_, y_, g_, n_splits, seed):
        splits = original(X_, y_, g_, n_splits, seed)
        for tr, te in splits:
            assert not set(g_[tr]) & set(g_[te]), "a group leaked across train and test"
        seen["ok"] = True
        return splits

    cv._split = spy
    try:
        out = cross_validate_model(model, X, y, ["a", "b"], n_splits=4, groups=groups, n_boot=20)
    finally:
        cv._split = original
    assert seen.get("ok") and out["grouped"] is True


def test_duplicates_inflate_ungrouped_cv_but_not_grouped():
    """Near-duplicate rows with random labels: ungrouped CV 'learns' them, grouped CV cannot."""
    rng = np.random.default_rng(1)
    n_groups, per_group = 60, 4
    groups = np.repeat(np.arange(n_groups), per_group)
    base = rng.normal(size=(n_groups, 6))
    y = np.repeat(rng.integers(0, 2, n_groups), per_group)       # labels unrelated to X
    X = np.repeat(base, per_group, axis=0) + rng.normal(scale=0.01, size=(len(y), 6))
    model = build_model("knn", {"n_neighbors": 1}, ImbalanceConfig.from_config(None), 0)
    plain = cross_validate_model(model, X, y, ["a", "b"], n_splits=5, n_boot=20)
    grouped = cross_validate_model(model, X, y, ["a", "b"], n_splits=5, groups=groups, n_boot=20)
    assert plain["pooled"]["accuracy"] > 0.9
    assert grouped["pooled"]["accuracy"] < 0.7


def test_drop_prefixes_removes_families():
    fvs = [{"video_name": "a", "activity": "x", "distortion": "d",
            "diag_laplacian_mean": 1.0, "meta_fps": 10.0, "app_lbp_mean_0": 0.1,
            "mean_speed_px_s": 3.0, "has_trajectory": 1.0, "n_tracks": 2},
           {"video_name": "b", "activity": "y", "distortion": "d",
            "diag_laplacian_mean": 2.0, "meta_fps": 10.0, "app_lbp_mean_0": 0.2,
            "mean_speed_px_s": 4.0, "has_trajectory": 0.0, "n_tracks": 0}]
    _, _, _, keys = ExperimentRunner._build_matrix(fvs, "activity", drop_prefixes=["diag_", "meta_", "app_"])
    assert keys == ["mean_speed_px_s"]
    _, _, _, keys_all = ExperimentRunner._build_matrix(fvs, "activity")
    assert "has_trajectory" not in keys_all and "n_tracks" not in keys_all


def _det(x, y, s=40):
    return make_bbox(x, y, s, s)


def test_tracker_records_frame_indices():
    trk = IoUHungarianTracker(iou_threshold=0.1)
    for f in range(5):
        trk.update([_det(10 + 2 * f, 10)])
    (t,) = trk.get_active_tracks()
    assert t["frames"] == [0, 1, 2, 3, 4] and len(t["frames"]) == len(t["trajectory"])


def test_interaction_uses_shared_frames_not_indices():
    # A walks frames 0..9 at x=0; B appears at frame 5 and is 500px away, then ends at frame 9.
    # Index-wise comparison would compare A@0..4 with B@5..9 differently than frame-wise.
    a = {"id": 0, "trajectory": [(0.0, 0.0)] * 10, "frames": list(range(10))}
    b = {"id": 1, "trajectory": [(500.0, 0.0)] * 5, "frames": list(range(5, 10))}
    feat = InteractionFeatures(min_trajectory_len=2).extract_track(
        a, all_tracks=[a, b], fps=10.0)
    assert feat["mean_pairwise_distance"] == pytest.approx(500.0)
    assert feat["proximity_duration_ratio"] == 0.0
    # only the 5 shared frames are compared
    b2 = {"id": 1, "trajectory": [(500.0, 0.0)] * 5, "frames": list(range(20, 25))}
    assert InteractionFeatures(min_trajectory_len=2).extract_track(
        a, all_tracks=[a, b2], fps=10.0) is None


def test_interaction_feature_names_match_output():
    f = InteractionFeatures(min_trajectory_len=2)
    a = {"id": 0, "trajectory": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], "frames": [0, 1, 2]}
    b = {"id": 1, "trajectory": [(5.0, 0.0), (6.0, 0.0), (8.0, 0.0)], "frames": [0, 1, 2]}
    out = f.extract_track(a, all_tracks=[a, b], fps=10.0)
    assert set(out) == set(f.feature_names())


def _write_video(path, n=60, fps=10.0, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 90, np.uint8)
        frame = np.clip(frame + rng.normal(0, 1, frame.shape), 0, 255).astype(np.uint8)
        x = 10 + 2 * i
        frame[40:80, x:x + 30] = 230          # one moving object
        if n - 8 <= i < n - 3:
            frame[5:25, 120:150] = 20         # a short-lived second object (5 frames near the end)
        w.write(frame)
    w.release()


def _runner(tmp_path, **kw):
    cfg = Config({
        "preprocessing": {"type": "standard", "frame_resize": [160, 120], "blur_kernel": [5, 5], "blur_sigma": 0},
        "detection": {"type": "mog2", "mog2": {"history": 10, "var_threshold": 16, "detect_shadows": False},
                      "morphology": {"close_iter": 1, "open_iter": 1}, "min_contour_area": 60},
        "tracking": {"type": "iou_hungarian", "max_disappeared": 30, "max_trajectory_len": 300,
                     "min_trajectory_len": 15, "iou_hungarian": {"iou_threshold": 0.1}},
        "optical_flow": {"lk_win_size": [15, 15], "lk_max_level": 2, "lk_criteria_eps": 0.03,
                         "lk_criteria_count": 10, "max_corners": 50, "corner_quality": 0.3, "corner_min_dist": 7},
        "features": {"modules": ["trajectory"], "aggregation": ["mean", "max", "std"]},
    })
    pre, det, trk, flow, ext, agg = PipelineBuilder.build_all(cfg)
    return PipelineRunner(pre, det, trk, flow, ext, agg, 15, **kw)


def test_short_track_filter_flag(tmp_path):
    vid = tmp_path / "v.avi"
    _write_video(vid)
    keep_all = _runner(tmp_path, filter_short_tracks=False).process_video(vid)
    filtered = _runner(tmp_path, filter_short_tracks=True).process_video(vid)
    assert min(len(t["trajectory"]) for t in filtered.tracks) >= 15
    assert len(keep_all.tracks) > len(filtered.tracks)       # the 4-frame blob only survives unfiltered


def test_speed_uses_effective_fps_under_frame_sampling(tmp_path):
    vid = tmp_path / "v.avi"
    _write_video(vid, n=80)
    full = _runner(tmp_path).process_video(vid, frame_sampling=1)
    sampled = _runner(tmp_path).process_video(vid, frame_sampling=2)
    def speed(r):
        key = "mean_mean_speed_px_s"
        return r.video_feature[key]
    # object moves 2 px/frame at 10 fps = 20 px/s whatever the sampling
    assert speed(full) == pytest.approx(speed(sampled), rel=0.25)


def test_constant_speed_track_gives_finite_shape_statistics():
    """scipy returns NaN for skew/kurtosis of constant data; SVM and KNN cannot take NaN."""
    assert safe_skew([2.0] * 8) == 0.0 and safe_kurtosis([2.0] * 8) == 0.0
    assert safe_skew([1.0, 2.0]) == 0.0
    assert safe_skew([1.0, 2.0, 9.0, 3.0]) != 0.0


def test_keep_prefixes_keeps_only_the_named_families():
    fvs = [{"activity": "a", "distortion": "x", "video_name": "v", "app_mean": 1.0,
            "diag_laplacian_mean": 2.0, "meta_fps": 10.0, "traj_speed_mean": 3.0},
           {"activity": "b", "distortion": "y", "video_name": "w", "app_mean": 4.0,
            "diag_laplacian_mean": 5.0, "meta_fps": 10.0, "traj_speed_mean": 6.0}]
    _, _, _, keys = ExperimentRunner._build_matrix(fvs, "activity", keep_prefixes=["app_", "diag_"])
    assert keys == ["app_mean", "diag_laplacian_mean"]
    _, _, _, keys = ExperimentRunner._build_matrix(fvs, "activity", keep_prefixes=["app_", "meta_"],
                                                   drop_prefixes=["meta_fps"])
    assert keys == ["app_mean"]
