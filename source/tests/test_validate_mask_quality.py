import numpy as np

import validate_mask_quality as v
from activity_detection.detection.adaptive_mog2 import AdaptiveMOG2Detector


def test_f_measure_edge_cases():
    empty = np.zeros((10, 10), np.uint8)
    full = np.full((10, 10), 255, np.uint8)
    assert v.mask_f_measure(empty, empty) == 1.0
    assert v.mask_f_measure(full, empty) == 0.0
    assert v.mask_f_measure(full, full) == 1.0


def test_boxes_to_mask_clips_and_skips_none():
    m = v.boxes_to_mask((20, 20), [None, {"x": -5, "y": 2, "w": 10, "h": 4}])
    assert m[2:6, 0:5].all() and m.sum() == 255 * 20


def test_video_frame_scores_runs_on_synthetic_frames():
    frames, boxes = [], []
    rng = np.random.default_rng(0)
    for i in range(50):
        f = np.full((120, 160, 3), 90, np.uint8)
        if i >= 15:                                   # background-only warm-up, then an object enters
            x = 5 + 3 * (i - 15)
            f[40:80, x:x + 30] = rng.integers(150, 255, (40, 30, 3))   # appearance changes every frame
            boxes.append({"x": x, "y": 40, "w": 30, "h": 40})
        else:
            boxes.append(None)
        frames.append(f)
    det = AdaptiveMOG2Detector(history=10, var_threshold=16, min_contour_area=60,
                               detect_shadows=False)
    pairs = v.video_frame_scores(frames, boxes, det)
    assert len(pairs) == 50 - (10 // 2 + 1)
    assert all(0 <= q <= 1 and 0 <= f <= 1 for q, f in pairs)
    with_object = [f for (_, f), i in zip(pairs, range(6, 50)) if i >= 16]
    assert np.mean(with_object) > 0.5                 # mask overlaps the GT box


def test_correlation_nan_when_constant():
    assert np.isnan(v.correlation([(0.5, 0.1)] * 5))
