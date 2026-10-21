"""Tests for the parallel video collection and the custom random forest."""
import cv2
import numpy as np
import pandas as pd

from activity_detection.config import load_config
from activity_detection.experiment.collector import VideoFeatureCollector, resolve_worker_count
from activity_detection.models.decision_tree import RandomForestCustom


def test_worker_count_auto_leaves_one_core(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert resolve_worker_count("auto", 100) == 7


def test_worker_count_never_exceeds_the_number_of_videos():
    assert resolve_worker_count(16, 3) == 3
    assert resolve_worker_count(0, 3) == 1
    assert resolve_worker_count(None, 3) == 1


def test_forest_is_the_same_for_any_number_of_jobs():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(150, 12)), rng.integers(0, 4, 150)
    serial = RandomForestCustom(n_trees=5, random_state=3, class_weight="balanced", n_jobs=1).fit(X, y)
    parallel = RandomForestCustom(n_trees=5, random_state=3, class_weight="balanced", n_jobs=2).fit(X, y)
    assert (serial.predict(X) == parallel.predict(X)).all()
    assert np.allclose(serial.feature_importances_, parallel.feature_importances_)


def test_forest_learns_a_separable_problem():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    y = (X[:, 2] > 0).astype(int)
    forest = RandomForestCustom(n_trees=10, random_state=0).fit(X, y)
    assert (forest.predict(X) == y).mean() > 0.9
    assert forest.feature_importances_.argmax() == 2


def _write_video(path, seed, n_frames=80):
    """An empty noisy scene for 15 frames, then a bright square that walks across it."""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(n_frames):
        frame = np.full((120, 160, 3), 30, np.uint8) + rng.integers(0, 3, (120, 160, 3), dtype=np.uint8)
        if i >= 15:
            x = 5 + (i - 15) * 2
            frame[40:80, x:x + 40] = 220
        writer.write(frame)
    writer.release()


def test_parallel_collection_gives_the_same_rows_as_sequential(tmp_path):
    names = [f"clip_{i}.avi" for i in range(4)]
    for i, name in enumerate(names):
        _write_video(tmp_path / name, seed=i)
    df = pd.DataFrame({"Name of Video Series": names,
                       "Activity": ["WL", "WL", "RK", "RK"],
                       "Distortion": ["Pri"] * 4})

    def collect(n_workers):
        cfg = load_config("group_a_classical", overrides={
            "paths": {"video_dir": str(tmp_path)},
            "experiment": {"n_workers": n_workers},
            "preprocessing": {"frame_resize": [160, 120]},
            "tracking": {"min_trajectory_len": 5},
        })
        rows = VideoFeatureCollector(cfg).collect(df)
        return [row["video_name"] for row in rows], rows

    seq_names, seq_rows = collect(1)
    par_names, par_rows = collect(2)
    assert seq_names == par_names == sorted(seq_names)
    assert len(seq_rows) >= 1
    numeric = [k for k, v in seq_rows[0].items() if isinstance(v, (int, float))]
    for a, b in zip(seq_rows, par_rows):
        assert np.allclose([a[k] for k in numeric], [b[k] for k in numeric], equal_nan=True)
