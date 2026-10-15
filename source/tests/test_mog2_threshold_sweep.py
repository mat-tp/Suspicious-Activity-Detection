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
mod.yield_by_distortion = sys.modules["sweep_common"].yield_by_distortion


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


def test_threshold_reaches_the_adaptive_detector_too():
    """Sweep override must land on the detector object, for fixed AND adaptive bases."""
    from activity_detection.pipeline.builder import PipelineBuilder
    for base in ("group_a_classical", "ablation_1_detector_only"):
        cfg = mod.setting_config(base, 25, "random_forest_sklearn", 5)
        det = PipelineBuilder.build_detector(cfg)
        assert det.var_threshold == 25, base
    assert cfg.detection.adaptive_mog2.threshold_floor == 8.0     # tau_min untouched by the sweep


def _write_run(tmp_path, name, rows, columns=("video", "true", "random_forest_sklearn")):
    d = tmp_path / name
    d.mkdir()
    with open(d / "predictions_activity.csv", "w") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    return {"value": 16, "run_dir": str(d)}


def test_common_subset_scores_only_shared_videos(tmp_path):
    a = _write_run(tmp_path, "a", [("v1", "x", "x"), ("v2", "y", "y"), ("v3", "x", "y")])
    b = _write_run(tmp_path, "b", [("v1", "x", "x"), ("v2", "y", "x")])
    out = mod.add_common_subset_scores([a, b], "activity", "random_forest_sklearn")
    assert out[0]["n_common"] == out[1]["n_common"] == 2
    assert out[0]["macro_f1_common"] == 1.0 and out[1]["macro_f1_common"] < 1.0


def test_common_subset_warns_when_model_column_is_missing(tmp_path):
    a = _write_run(tmp_path, "a", [("v1", "x", "x")], columns=("video", "true", "other_model"))
    with pytest.warns(UserWarning, match="not in"):
        out = mod.add_common_subset_scores([a], "activity", "random_forest_sklearn")
    assert "macro_f1_common" not in out[0]


def test_common_subset_warns_when_predictions_file_is_missing(tmp_path):
    with pytest.warns(UserWarning, match="does not exist"):
        mod.add_common_subset_scores([{"value": 16, "run_dir": str(tmp_path)}], "activity", "m")


def test_yield_by_distortion_reads_videos_attempted(tmp_path):
    (tmp_path / "videos_attempted.csv").write_text(
        "video,activity,distortion,status\n"
        "a,walk,Focus,ok\nb,walk,Focus,failed: no_tracks\nc,walk,Pristine,ok\n")
    assert mod.yield_by_distortion(tmp_path) == {"yield_Focus": 0.5, "yield_Pristine": 1.0}
    assert mod.yield_by_distortion(tmp_path / "missing") == {}


def test_plot_and_csv_include_common_and_per_distortion_columns(tmp_path):
    rows = []
    for v, f1, yf in [(12, 0.60, 0.80), (16, 0.65, 0.90), (25, 0.62, 0.85)]:
        rows.append({"value": v, "macro_f1": f1, "macro_f1_low": f1 - .03, "macro_f1_high": f1 + .03,
                     "video_yield": 0.9, "macro_f1_common": f1 - .01, "yield_Focus": yf,
                     "yield_Pristine": 1.0, "run_dir": "x"})
    mod.write_rows(tmp_path / "r.csv", rows)
    header = (tmp_path / "r.csv").read_text().splitlines()[0].split(",")
    assert "macro_f1_common" in header and "yield_Focus" in header and header[-1] == "run_dir"
    pytest.importorskip("matplotlib")
    mod.plot({"group_a_classical": rows}, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000
