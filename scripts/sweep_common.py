"""Helpers shared by the sweep scripts (mog2_threshold_sweep.py, parameter_ablation.py)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from activity_detection.paths import EXPERIMENTS_DIR

# Sections that do not influence the extracted feature vectors; everything else does.
CONFIG_SECTIONS_IGNORED_BY_FEATURES = ("models", "training", "imbalance", "experiment",
                                       "targets", "smoothing")
# These are the files that are copied together with a cached feature set.
FILES_SAVED_WITH_FEATURES = ("video_features.csv", "videos_attempted.csv",
                             "adaptive_mog2_diagnostics.json", "detector_stats.json")


def read_rows(path: Path):
    """Rows of a results CSV, or [] when the file does not exist yet."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def yield_by_distortion(run_dir):
    """Share of attempted videos that produced features, as ``{"yield_<Distortion>": share}``.

    It reads ``videos_attempted.csv``, so the denominator is the videos the run really tried. It returns
    ``{}`` when the file does not exist.
    """
    path = Path(run_dir) / "videos_attempted.csv"
    if not path.exists():
        return {}
    succeeded, attempted = {}, {}
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            distortion = str(record["distortion"])
            attempted[distortion] = attempted.get(distortion, 0) + 1
            succeeded[distortion] = succeeded.get(distortion, 0) + (record["status"] == "ok")
    return {f"yield_{d}": succeeded[d] / attempted[d] for d in sorted(attempted)}


def feature_cache_dir(config, use_sample: bool) -> Path:
    """Folder keyed by everything that determines the features (not by model or target)."""
    relevant = {k: v for k, v in config.to_dict().items()
                if k not in CONFIG_SECTIONS_IGNORED_BY_FEATURES}
    relevant["seed"] = config.experiment.get("seed")      # decides which videos --sample picks
    key = hashlib.sha1(json.dumps([relevant, use_sample], sort_keys=True, default=str).encode())
    return EXPERIMENTS_DIR / "_feature_cache" / key.hexdigest()[:16]


def collect_features_cached(runner, runner_class, config, dataset, use_sample: bool):
    """Process the videos once per distinct pipeline setting.

    Sweeps that differ only in the target or the classifier share the same features, so the later ones load
    them from disk. The files that describe the run (attempted videos, detector diagnostics) are copied into
    the new run folder.
    """
    can_cache = hasattr(runner_class, "load_video_features") and getattr(runner, "run_dir", None)
    if not can_cache:
        return runner.collect_video_features(dataset, use_sample=use_sample)
    cache_dir = feature_cache_dir(config, use_sample)
    if (cache_dir / "video_features.csv").exists():
        print(f"  Reusing cached features: {cache_dir}")
        for name in FILES_SAVED_WITH_FEATURES:
            if (cache_dir / name).exists():
                shutil.copy(cache_dir / name, Path(runner.run_dir) / name)
        return runner_class.load_video_features(cache_dir / "video_features.csv")
    features = runner.collect_video_features(dataset, use_sample=use_sample)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES_SAVED_WITH_FEATURES:
        if (Path(runner.run_dir) / name).exists():
            shutil.copy(Path(runner.run_dir) / name, cache_dir / name)
    return features


def run_setting(config, experiment_name, target, use_sample, runner_class):
    """Run one setting end to end and return the numbers for one CSV row."""
    runner = runner_class(config, experiment_name=experiment_name)
    dataset = runner.load_dataset()
    n_videos_in_dataset = len(dataset)
    features = collect_features_cached(runner, runner_class, config, dataset, use_sample)
    results = runner.train_and_evaluate(features, target_col=target)

    scored = next((r for r in results.values() if "cv" in r), None)
    if scored is None:
        raise RuntimeError(f"No model finished for {experiment_name}: {results}")
    cv = scored["cv"]
    row = {
        "macro_f1": cv["pooled"]["macro_f1"],
        "macro_f1_low": cv["ci95"]["macro_f1"][0],
        "macro_f1_high": cv["ci95"]["macro_f1"][1],
        "accuracy": cv["pooled"]["accuracy"],
        "balanced_accuracy": cv["pooled"]["balanced_accuracy"],
        "video_yield": len(features) / max(n_videos_in_dataset, 1),
        "trigger_rate": "", "accept_rate": "", "run_dir": str(runner.run_dir),
    }
    row.update(yield_by_distortion(runner.run_dir))
    diagnostics_file = Path(runner.run_dir) / "adaptive_mog2_diagnostics.json"
    if diagnostics_file.exists():
        diagnostics = json.loads(diagnostics_file.read_text())
        if diagnostics:
            row["trigger_rate"] = sum(d["adaptive_trigger_rate"] for d in diagnostics) / len(diagnostics)
            row["accept_rate"] = sum(d["rerun_accept_rate"] for d in diagnostics) / len(diagnostics)
    return row
