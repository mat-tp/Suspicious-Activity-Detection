"""Tests for ViBe, PBAS, SuBSENSE and the MLP."""
import importlib.util

import cv2
import numpy as np
import pytest

from activity_detection.config import load_config
from activity_detection.detection.pbas import PBASDetector
from activity_detection.detection.subsense import SuBSENSEDetector
from activity_detection.detection.vibe import ViBeDetector
from activity_detection.evaluation.cross_validation import cross_validate_model
from activity_detection.models.imbalance import ImbalanceConfig, build_model
from activity_detection.pipeline.builder import PipelineBuilder

H, W, ENTRY = 240, 320, 30


def scene(n_frames=90, seed=0):
    """Noisy empty background; from frame ENTRY a bright box moves across it."""
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for i in range(n_frames):
        frame = np.full((H, W), 110, np.uint8) + rng.integers(0, 10, (H, W), dtype=np.uint8)
        box = None
        if i >= ENTRY:
            x = 20 + 4 * (i - ENTRY)
            cv2.rectangle(frame, (x, 90), (x + 50, 160), 235, -1)
            box = (x, 90, 50, 70)
        frames.append(frame)
        truth.append(box)
    return frames, truth


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


@pytest.mark.parametrize("detector_class", [ViBeDetector, PBASDetector])
def test_finds_the_moving_object_and_stays_quiet_on_empty_scenes(detector_class):
    detector = detector_class(min_contour_area=300)
    frames, truth = scene()
    hits, quiet_frames = 0, 0
    for i, (frame, box) in enumerate(zip(frames, truth)):
        found = [(b["x"], b["y"], b["w"], b["h"]) for b in detector.detect(frame)]
        if box is None and i >= 10:
            quiet_frames += not found
        elif box is not None and i >= ENTRY + 5:
            hits += any(iou(box, f) > 0.5 for f in found)
    n_after_entry = len(frames) - ENTRY - 5
    assert hits >= 0.9 * n_after_entry
    assert quiet_frames >= 0.9 * (ENTRY - 10)


@pytest.mark.parametrize("detector_class", [ViBeDetector, PBASDetector])
def test_same_seed_gives_same_result_and_reset_restarts(detector_class):
    frames, _ = scene(50)
    first = [detector_class().detect(f) for f in frames[:1]]       # model starts empty
    detector = detector_class()
    run_a = [len(detector.detect(f)) for f in frames]
    detector.reset()
    run_b = [len(detector.detect(f)) for f in frames]
    assert run_a == run_b and first is not None


def test_subsense_reports_missing_package_clearly():
    try:
        import pybgs  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pybgs"):
            SuBSENSEDetector()


@pytest.mark.parametrize("name", ["vibe", "pbas"])
def test_detector_configs_change_only_the_detector(name):
    base, other = load_config("group_a_classical"), load_config(f"detector_{name}")
    assert other.detection.type == name
    assert other.preprocessing.type == base.preprocessing.type
    assert other.features.modules == base.features.modules
    assert other.tracking.type == base.tracking.type
    detector = PipelineBuilder.build_detector(other)
    assert detector.min_contour_area == base.detection.min_contour_area
    assert detector.morph_close_iter == base.detection.morphology.close_iter


@pytest.mark.skipif(
    importlib.util.find_spec("imblearn") is None,
    reason="imbalanced-learn not installed",
)
def test_mlp_trains_with_class_weights_and_resampling():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [25, 90, 90])
    X = rng.normal(size=(len(y), 8)) + y[:, None]
    for imbalance in (ImbalanceConfig(class_weight="balanced"),
                      ImbalanceConfig(class_weight="none", resampling="smote")):
        model = build_model("mlp", {"hidden_layer_sizes": [16], "max_iter": 300, "early_stopping": False}, imbalance, 0)
        cv = cross_validate_model(model, X, y, ["a", "b", "c"], n_splits=3, n_boot=20)
        assert cv["pooled"]["macro_f1"] > 0.5
