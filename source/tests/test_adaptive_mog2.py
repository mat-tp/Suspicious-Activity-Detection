"""Tests for the adaptive MOG2 detector and the ablation configs.

Run from the source/ folder:   python -m pytest tests
"""
import cv2
import numpy as np

from activity_detection.config import load_config
from activity_detection.detection.adaptive_mog2 import AdaptiveMOG2Detector
from activity_detection.detection.mog2 import MOG2Detector
from activity_detection.pipeline.builder import PipelineBuilder

SETTINGS = dict(var_threshold=25, detect_shadows=False)


def make_frames(n=150):
    """Noisy grey background, a moving bright box, and a blurred frame every 7th."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        frame = np.full((240, 320), 100, np.uint8) + rng.integers(0, 12, (240, 320), dtype=np.uint8)
        cv2.rectangle(frame, (10 + 2 * i, 80), (60 + 2 * i, 160), 230, -1)
        if i % 7 == 0:
            frame = cv2.GaussianBlur(frame, (0, 0), 4)
        frames.append(frame)
    return frames


def boxes_per_frame(detector, frames):
    return [[(b["x"], b["y"], b["w"], b["h"]) for b in detector.detect(f)] for f in frames]


def test_no_adaptation_means_identical_to_plain_mog2():
    frames = make_frames()
    plain = boxes_per_frame(MOG2Detector(**SETTINGS), frames)
    adaptive = boxes_per_frame(AdaptiveMOG2Detector(quality_threshold=0.0, **SETTINGS), frames)
    assert plain == adaptive


def test_relaxed_look_does_not_change_the_background_model():
    frames = make_frames()
    plain = cv2.createBackgroundSubtractorMOG2(20, 25, False)
    peeked = cv2.createBackgroundSubtractorMOG2(20, 25, False)
    for i, frame in enumerate(frames):
        a = plain.apply(frame, learningRate=-1)
        b = peeked.apply(frame, learningRate=-1)
        if i >= 10:
            peeked.setVarThreshold(8)
            peeked.apply(frame, learningRate=0)      # the "second look"
            peeked.setVarThreshold(25)
        assert (a == b).all()


def test_decision_log_has_one_row_per_frame():
    frames = make_frames(60)
    detector = AdaptiveMOG2Detector(log_decisions=True, **SETTINGS)
    boxes_per_frame(detector, frames)
    assert len(detector.decisions) == len(frames)
    assert detector.get_diagnostics()["n_frames"] == len(frames)


def test_reset_clears_state():
    detector = AdaptiveMOG2Detector(log_decisions=True, **SETTINGS)
    boxes_per_frame(detector, make_frames(30))
    detector.reset()
    assert detector.decisions == []
    assert detector.get_diagnostics()["n_frames"] == 0


def test_group_i_yaml_parameters_reach_the_detector():
    cfg = load_config("group_i_adaptive_mog2", overrides={
        "detection": {"adaptive_mog2": {"quality_threshold": 0.55, "mask_weight": 0.3}}})
    detector = PipelineBuilder.build_detector(cfg)
    assert detector.quality_threshold == 0.55
    assert detector.mask_weight == 0.3


def factors(name):
    cfg = load_config(name)
    return {
        "detector": cfg.detection.type,
        "preprocessing": cfg.preprocessing.type,
        "features": tuple(cfg.features.modules),
        "multifamily": bool(cfg.features.get("enable_multifamily", False)),
    }


def changed(a, b):
    fa, fb = factors(a), factors(b)
    return sorted(k for k in fa if fa[k] != fb[k])


def test_each_ablation_changes_exactly_one_factor():
    assert changed("group_a_classical", "ablation_1_detector_only") == ["detector"]
    assert changed("group_a_classical", "ablation_2_preprocessing_only") == ["preprocessing"]
    assert changed("group_a_classical", "ablation_3_features_only") == ["features", "multifamily"]
    assert changed("group_i_adaptive_mog2", "ablation_4_full_pipeline_fixed_mog2") == ["detector"]


def _background_after(detector, frames):
    for frame in frames:
        detector.detect(frame)
    return detector.mog2.getBackgroundImage()


def _background_gap(detector, reference_background, frames):
    """(largest grey-level difference, share of differing pixels) against a reference."""
    background = _background_after(detector, frames)
    difference = np.abs(reference_background.astype(int) - background.astype(int))
    return int(difference.max()), float((difference > 0).mean())


def test_adaptive_retry_never_advances_the_background_model_twice():
    """A quality threshold above 1 forces a relaxed retry on every frame.

    With ``update_once=True`` the retry uses learning rate 0, so the background stays close to plain MOG2's
    (OpenCV may reorder mixture components, which moves a few pixels by one grey level). The old double
    update changes a large share of the image.
    """
    frames = make_frames()
    plain_background = _background_after(MOG2Detector(**SETTINGS), frames)

    always_retry = AdaptiveMOG2Detector(quality_threshold=1.5, update_once=True, **SETTINGS)
    largest_gap, share_differing = _background_gap(always_retry, plain_background, frames)
    assert always_retry.get_diagnostics()["adaptive_trigger_rate"] > 0.5   # retries did happen
    assert largest_gap <= 1 and share_differing < 1e-3

    double_update = AdaptiveMOG2Detector(quality_threshold=1.5, update_once=False, **SETTINGS)
    _, share_double = _background_gap(double_update, plain_background, frames)
    assert share_double > 0.05
