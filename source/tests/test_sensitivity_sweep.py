"""Tests for the sensitivity sweep script, using a fake runner (no videos needed)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sensitivity_sweep.py"
spec = importlib.util.spec_from_file_location("sensitivity_sweep", SCRIPT)
sweep_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_module)


class FakeRunner:
    """Stands in for ExperimentRunner and records which settings it was given."""
    seen = []

    def __init__(self, cfg, experiment_name=None):
        FakeRunner.seen.append(cfg.detection.adaptive_mog2.to_dict())
        self.run_dir = Path("/tmp")

    def load_dataset(self):
        return [0] * 10

    def collect_video_features(self, df, use_sample=False):
        return [0] * 9

    def train_and_evaluate(self, features, target_col):
        return {"random_forest_sklearn": {"cv": {
            "pooled": {"macro_f1": 0.7, "accuracy": 0.8, "balanced_accuracy": 0.75},
            "ci95": {"macro_f1": [0.65, 0.75]}}}}


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "EXPERIMENTS_DIR", tmp_path)
    FakeRunner.seen = []


def args(*extra):
    return sweep_module.parse_args(["--params", "q", "tau_min", *extra])


def test_plan_runs_the_default_once_and_skips_default_values():
    grid = {"q": [0.3, 0.4, 0.5]}
    plan = sweep_module.make_plan(grid, {"quality_threshold": 0.4})
    assert plan == [("default", None), ("q", 0.3), ("q", 0.5)]


def test_only_the_swept_parameter_changes():
    cfg = sweep_module.setting_config("ablation_1_detector_only", "q", 0.55, "knn", 5)
    default = sweep_module.setting_config("ablation_1_detector_only", "default", None, "knn", 5)
    a, b = cfg.detection.adaptive_mog2.to_dict(), default.detection.adaptive_mog2.to_dict()
    assert a["quality_threshold"] == 0.55
    assert {k: v for k, v in a.items() if k != "quality_threshold"} == \
           {k: v for k, v in b.items() if k != "quality_threshold"}


def test_sweep_writes_a_row_per_setting_and_resumes(tmp_path):
    rows = sweep_module.sweep(args("--values", "q=0.3,0.5", "tau_min=5"), FakeRunner)
    assert len(rows) == 1 + 2 + 1          # default + 2 q values + 1 tau_min value
    first_run = len(FakeRunner.seen)
    assert any(s["quality_threshold"] == 0.3 for s in FakeRunner.seen)

    FakeRunner.seen = []
    sweep_module.sweep(args("--values", "q=0.3,0.5", "tau_min=5", "--resume"), FakeRunner)
    assert FakeRunner.seen == [] and first_run == 4     # nothing was rerun


def test_non_adaptive_config_is_rejected():
    with pytest.raises(SystemExit):
        sweep_module.sweep(sweep_module.parse_args(["--base-config", "group_a_classical"]), FakeRunner)


def test_unknown_parameter_in_values_is_rejected():
    with pytest.raises(SystemExit):
        sweep_module.build_grid(["q"], ["zeta=1,2"])
