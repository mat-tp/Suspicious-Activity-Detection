"""Tests for the var_threshold ablation script (fake runner, no videos needed)."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mog2_threshold_sweep", SCRIPTS / "mog2_threshold_sweep.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeRunner:
    seen = []

    def __init__(self, cfg, experiment_name=None):
        FakeRunner.seen.append((cfg.detection.type, cfg.detection.mog2.var_threshold))
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
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPERIMENTS_DIR", tmp_path)
    FakeRunner.seen = []


def test_adaptive_skips_values_at_or_below_tau_r():
    cfg = mod.load_config("ablation_1_detector_only")      # tau_r = 10
    run, skipped = mod.make_plan(cfg, [8, 10, 12, 16])
    assert run == [12, 16] and skipped == [8, 10]


def test_fixed_runs_every_value():
    cfg = mod.load_config("group_a_classical")
    assert mod.make_plan(cfg, [8, 16, 25])[0] == [8, 16, 25]


def test_other_detectors_rejected():
    with pytest.raises(SystemExit):
        mod.make_plan(mod.load_config("detector_vibe"), [16])


def test_threshold_override_reaches_config_and_only_it():
    base = mod.load_config("group_a_classical")
    cfg = mod.setting_config("group_a_classical", 25, "random_forest_sklearn", 5)
    assert cfg.detection.mog2.var_threshold == 25
    assert cfg.detection.mog2.history == base.detection.mog2.history
    assert cfg.tracking.to_dict() == base.tracking.to_dict()


def test_sweep_runs_both_detectors_and_resumes():
    a = mod.parse_args(["--values", "12", "16", "25"])
    res = mod.sweep(a, FakeRunner)
    assert set(res) == {"group_a_classical", "ablation_1_detector_only"}
    assert len(FakeRunner.seen) == 6
    FakeRunner.seen = []
    mod.sweep(mod.parse_args(["--values", "12", "16", "25", "--resume"]), FakeRunner)
    assert FakeRunner.seen == []


def test_default_is_the_literature_value():
    assert mod.load_config("group_a_classical").detection.mog2.var_threshold == mod.LITERATURE_VALUE
