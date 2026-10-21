"""Tests for the one-at-a-time parameter ablation, using a fake runner (no videos needed)."""
from pathlib import Path

import pytest

import parameter_ablation as ablation


class FakeRunner:
    """Stands in for ExperimentRunner and records the config of every setting."""

    seen = []
    collect_calls = 0

    def __init__(self, cfg, experiment_name=None):
        FakeRunner.seen.append(cfg)
        self.cfg = cfg
        self.run_dir = Path("/tmp")

    def load_dataset(self):
        return [0] * 10

    def collect_video_features(self, df, use_sample=False):
        FakeRunner.collect_calls += 1
        return [{"video_name": "v"}] * 9

    @staticmethod
    def load_video_features(path):
        return [{"video_name": "v"}] * 9

    def train_and_evaluate(self, features, target_col):
        return {self.cfg.models.active[0]["name"]: {
            "feature_count": 5,
            "cv": {"pooled": {"macro_f1": 0.7, "accuracy": 0.8, "balanced_accuracy": 0.75},
                   "ci95": {"macro_f1": [0.65, 0.75]}}}}


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "EXPERIMENTS_DIR", tmp_path)
    FakeRunner.seen = []
    FakeRunner.collect_calls = 0


def parse(*extra):
    return ablation.parse_args(list(extra))


def test_every_config_path_exists_in_the_reference_config():
    """An ablated parameter that is not in the config would silently do nothing."""
    reference = ablation.load_config(ablation.REFERENCE_CONFIG).to_dict()
    for p in ablation.PARAMETER_TABLE:
        if p.stage == "pipeline":
            assert ablation.get_by_path(reference, p.config_path, "MISSING") != "MISSING", p.name


def test_plan_runs_reference_once_and_skips_reference_values():
    reference = ablation.load_config(ablation.REFERENCE_CONFIG).to_dict()
    params = ablation.select_parameters(["adaptive"], ["q_quality_threshold"])
    plan = ablation.make_plan(params, reference, "random_forest_sklearn")
    values = [v for p, v, _ in plan if p is not None]
    assert 0.4 not in values and values == [0.2, 0.3, 0.5, 0.6]
    assert plan[0][0] is None


def test_companion_parameter_reruns_the_reference_value():
    reference = ablation.load_config(ablation.REFERENCE_CONFIG).to_dict()
    params = ablation.select_parameters(["tracker"], ["min_trajectory_len_filtered"])
    plan = ablation.make_plan(params, reference, "random_forest_sklearn")
    assert [v for p, v, _ in plan if p is not None] == [5, 15, 30, 45]


def test_only_the_ablated_parameter_changes():
    base = ablation.load_config(ablation.REFERENCE_CONFIG).to_dict()
    parameter = next(p for p in ablation.PARAMETER_TABLE if p.name == "iou_threshold")
    overrides = ablation.pipeline_overrides(parameter, 0.5, 5, 0, "knn")
    changed = ablation.load_config(ablation.REFERENCE_CONFIG, overrides=overrides).to_dict()
    assert changed["tracking"]["iou_hungarian"]["iou_threshold"] == 0.5
    changed["tracking"]["iou_hungarian"]["iou_threshold"] = base["tracking"]["iou_hungarian"]["iou_threshold"]
    for ignored in ("models", "dataset"):
        changed.pop(ignored), base.pop(ignored)
    assert changed == base


def test_model_stage_override_only_touches_the_tuned_parameter():
    base = ablation.load_config(ablation.REFERENCE_CONFIG).to_dict()
    parameter = next(p for p in ablation.PARAMETER_TABLE if p.name == "svm_rbf.C")
    overrides = ablation.model_overrides(parameter, 100, base, 5)
    (model,) = overrides["models"]["active"]
    assert model["name"] == "svm_rbf" and model["params"]["C"] == 100
    assert model["params"]["gamma"] == "scale"


def test_pipeline_settings_reprocess_videos_but_model_settings_reuse_features():
    rows = ablation.run_ablation(
        parse("--groups", "adaptive", "classifier", "--only", "q_quality_threshold", "svm_rbf.C",
              "--targets", "activity", "distortion"), FakeRunner)
    n_pipeline = 1 + 4        # reference + four q values
    assert FakeRunner.collect_calls == n_pipeline
    assert {r["stage"] for r in rows} == {"pipeline", "model"}
    assert {r["target"] for r in rows} == {"activity", "distortion"}


def test_resume_skips_finished_settings():
    args = ("--groups", "adaptive", "--only", "w_mask_weight", "--targets", "activity")
    first = ablation.run_ablation(parse(*args), FakeRunner)
    calls = FakeRunner.collect_calls
    ablation.run_ablation(parse(*args, "--resume"), FakeRunner)
    assert FakeRunner.collect_calls == calls and len(first) == calls
