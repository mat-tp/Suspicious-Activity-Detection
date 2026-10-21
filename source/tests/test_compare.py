"""Tests for comparing experiments."""

import numpy as np
import pandas as pd
import pytest

from activity_detection.config import load_config
from activity_detection.evaluation.compare import (
    compare_many,
    Pair,
    compare_runs,
    find_run_dir,
    holm_adjust,
    load_run,
    paired_bootstrap,
    scores,
)
from activity_detection.evaluation.report import (
    write_attempted_videos,
    write_predictions,
)
import activity_detection.experiment.runner as runner_module
from activity_detection.experiment.runner import ExperimentRunner

CLASSES = ["Pri", "Exp", "Fo"]
N = 600


def make_run(root, name, true, predictions, run="run_001", model="lightgbm"):
    """Write a fake run the way the real runner does. predictions[i] None = no prediction."""
    directory = root / name / run
    directory.mkdir(parents=True)
    videos = [f"v{i:04d}.avi" for i in range(N)]
    groups = {("Walking", c): [v for v, t in zip(videos, true) if t == c] for c in CLASSES}
    ok = [v for v, p in zip(videos, predictions) if p is not None]
    write_attempted_videos(directory / "videos_attempted.csv", groups, ok, {})
    kept = [i for i, p in enumerate(predictions) if p is not None]
    write_predictions(directory / "predictions_distortion.csv", [videos[i] for i in kept],
                      [true[i] for i in kept], {model: [predictions[i] for i in kept]})
    return directory


def noisy(true, accuracy, rng, missing=0.0):
    """Predictions that are right with the given probability; some can be missing."""
    out = []
    for t in true:
        if rng.random() < missing:
            out.append(None)
        elif rng.random() < accuracy:
            out.append(t)
        else:
            out.append(str(rng.choice([c for c in CLASSES if c != t])))
    return out


@pytest.fixture
def truth():
    rng = np.random.default_rng(0)
    return [str(c) for c in rng.choice(CLASSES, N, p=[0.1, 0.45, 0.45])]


def load(root, name):
    return load_run(name, "distortion", "lightgbm", experiments_dir=root)


def test_identical_runs_show_no_difference(tmp_path, truth):
    pred = noisy(truth, 0.8, np.random.default_rng(1))
    make_run(tmp_path, "a", truth, pred)
    make_run(tmp_path, "b", truth, pred)
    r = compare_runs(load(tmp_path, "a"), load(tmp_path, "b"), "distortion", "all", n_boot=200)
    assert r["metrics"]["macro_f1"]["diff"] == 0
    assert r["metrics"]["accuracy"]["verdict"] == "interval includes 0"
    assert r["mcnemar"]["p_value"] == 1.0


def test_a_clearly_better_run_has_an_interval_above_zero(tmp_path, truth):
    rng = np.random.default_rng(2)
    make_run(tmp_path, "a", truth, noisy(truth, 0.65, rng))
    make_run(tmp_path, "b", truth, noisy(truth, 0.90, rng))
    r = compare_runs(load(tmp_path, "a"), load(tmp_path, "b"), "distortion", "all", n_boot=300)
    acc = r["metrics"]["accuracy"]
    assert acc["ci_low"] > 0 and acc["verdict"] == "B higher"
    assert r["mcnemar"]["p_value"] < 0.001
    assert acc["ci_low"] <= acc["diff"] <= acc["ci_high"]


def test_lost_videos_count_against_an_experiment_only_when_universe_is_all(tmp_path, truth):
    """Same accuracy on the videos it handles, but A loses half of them."""
    rng = np.random.default_rng(3)
    make_run(tmp_path, "a", truth, noisy(truth, 0.8, rng, missing=0.5))
    make_run(tmp_path, "b", truth, noisy(truth, 0.8, rng, missing=0.0))
    a, b = load(tmp_path, "a"), load(tmp_path, "b")
    common = compare_runs(a, b, "distortion", "common", n_boot=300)
    everything = compare_runs(a, b, "distortion", "all", n_boot=300)
    assert common["metrics"]["accuracy"]["verdict"] == "interval includes 0"
    assert everything["metrics"]["accuracy"]["verdict"] == "B higher"
    assert everything["counts"]["videos"] == N and everything["counts"]["evaluated_a"] < N * 0.6
    assert common["counts"]["videos"] < N


def test_missing_prediction_is_a_miss_in_per_class_scores():
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 2, 1, 1])        # index 2 = no prediction (k = 2)
    s = scores(true, pred, 2)
    assert s["accuracy"] == 0.75
    assert s["recall"][0] == 0.5 and s["recall"][1] == 1.0
    assert np.isclose(s["f1"][0], 2 * 1.0 * 0.5 / 1.5)


def test_holm_adjustment():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == [0.5]
    assert all(adj >= p for adj, p in zip(holm_adjust([0.2, 0.001, 0.04]), [0.2, 0.001, 0.04]))


def test_several_candidates_get_holm_adjusted_p_values(tmp_path, truth):
    rng = np.random.default_rng(4)
    make_run(tmp_path, "ref", truth, noisy(truth, 0.7, rng))
    for name, acc in [("c1", 0.75), ("c2", 0.9)]:
        make_run(tmp_path, name, truth, noisy(truth, acc, rng))
    ref = load(tmp_path, "ref")
    results = compare_many(ref, [load(tmp_path, "c1"), load(tmp_path, "c2")], "distortion", "all", n_boot=100)
    for r in results:
        assert r["mcnemar"]["p_holm"] >= r["mcnemar"]["p_value"]


def test_latest_run_with_predictions_is_used(tmp_path, truth):
    pred = noisy(truth, 0.8, np.random.default_rng(5))
    make_run(tmp_path, "a", truth, pred, run="run_001")
    newer = make_run(tmp_path, "a", truth, pred, run="run_002")
    (tmp_path / "a" / "run_003").mkdir()             # newest run, but it has no predictions
    assert find_run_dir("a", "distortion", tmp_path) == newer


def test_old_runs_without_predictions_give_a_clear_error(tmp_path):
    (tmp_path / "old" / "run_001").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run again|predictions_activity"):
        find_run_dir("old", "activity", tmp_path)


def test_unknown_model_is_reported(tmp_path, truth):
    make_run(tmp_path, "a", truth, noisy(truth, 0.8, np.random.default_rng(6)))
    with pytest.raises(KeyError, match="no model 'svm_rbf'"):
        load_run("a:svm_rbf", "distortion", "lightgbm", experiments_dir=tmp_path)


def test_experiment_runner_saves_one_prediction_per_video(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "EXPERIMENTS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    features = []
    for ci, name in enumerate(CLASSES):
        for i in range(40):
            row = {f"f{j}": float(rng.normal(ci, 1)) for j in range(5)}
            row.update(activity="Walking", distortion=name, video_name=f"{name}_{i}.avi")
            features.append(row)
    cfg = load_config("group_a_classical", overrides={
        "models": {"active": [{"name": "knn"}]}, "training": {"n_bootstrap": 20}})
    runner = ExperimentRunner(cfg, experiment_name="saved")
    runner.train_and_evaluate(features, "distortion")
    saved = pd.read_csv(runner.run_dir / "predictions_distortion.csv")
    assert list(saved.columns) == ["video", "true", "knn"]
    assert len(saved) == len(features) and saved["video"].is_unique


def test_cluster_bootstrap_gives_wider_interval_than_video_bootstrap():
    """Copies of one recording carry the same information, so resampling recordings
    must not give a narrower interval than resampling videos."""
    rng = np.random.default_rng(0)
    n_rec, copies = 40, 3
    true_r = rng.integers(0, 2, n_rec)
    a_r = np.where(rng.random(n_rec) < 0.7, true_r, 1 - true_r)
    b_r = np.where(rng.random(n_rec) < 0.8, true_r, 1 - true_r)

    def rep(v):
        return np.repeat(v, copies)

    groups = np.repeat(np.arange(n_rec), copies)
    base = dict(classes=["x", "y"], true=rep(true_r), pred_a=rep(a_r), pred_b=rep(b_r), counts={})

    def width(d):
        return np.subtract(*np.percentile(d["accuracy"], [97.5, 2.5]))

    plain = paired_bootstrap(Pair(**base), 500, 1)
    clustered = paired_bootstrap(Pair(**base, groups=groups), 500, 1)
    assert width(clustered) > width(plain) * 0.9
