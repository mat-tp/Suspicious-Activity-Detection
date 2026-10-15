"""Feature caching in the sweep helpers: the second target must not reprocess videos."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sweep_common  # noqa: E402

from activity_detection.config import load_config  # noqa: E402


class CountingRunner:
    collected = 0

    def __init__(self, run_dir):
        self.run_dir = run_dir

    def collect_video_features(self, dataset, use_sample=False):
        CountingRunner.collected += 1
        features = [{"video_name": "a", "x": 1.0}, {"video_name": "b", "x": 2.0}]
        pd.DataFrame(features).to_csv(Path(self.run_dir) / "video_features.csv", index=False)
        (Path(self.run_dir) / "videos_attempted.csv").write_text("video,status\na,ok\nb,ok\n")
        return features

    @staticmethod
    def load_video_features(path):
        return pd.read_csv(path).to_dict("records")


def test_second_call_with_same_pipeline_reuses_features(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_common, "EXPERIMENTS_DIR", tmp_path)
    CountingRunner.collected = 0
    config = load_config("group_a_classical")
    first_dir, second_dir = tmp_path / "run1", tmp_path / "run2"
    first_dir.mkdir(); second_dir.mkdir()

    sweep_common.collect_features_cached(CountingRunner(first_dir), CountingRunner, config, [], False)
    second = sweep_common.collect_features_cached(CountingRunner(second_dir), CountingRunner, config, [], False)

    assert CountingRunner.collected == 1
    assert [row["video_name"] for row in second] == ["a", "b"]
    assert (second_dir / "videos_attempted.csv").exists()      # run description is carried over


def test_a_changed_pipeline_setting_is_not_served_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_common, "EXPERIMENTS_DIR", tmp_path)
    CountingRunner.collected = 0
    base = load_config("group_a_classical")
    changed = load_config("group_a_classical", overrides={"detection": {"mog2": {"var_threshold": 40}}})
    for index, config in enumerate((base, changed)):
        run_dir = tmp_path / f"run{index}"
        run_dir.mkdir()
        sweep_common.collect_features_cached(CountingRunner(run_dir), CountingRunner, config, [], False)
    assert CountingRunner.collected == 2


def test_classifier_or_target_changes_do_not_invalidate_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_common, "EXPERIMENTS_DIR", tmp_path)
    base = load_config("group_a_classical")
    other_model = load_config("group_a_classical", overrides={"models": {"active": [{"name": "knn", "params": {}}]}})
    assert sweep_common.feature_cache_dir(base, False) == sweep_common.feature_cache_dir(other_model, False)
