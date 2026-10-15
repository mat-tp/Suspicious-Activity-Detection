"""Inspect the experiment outputs and fail loudly if any required result is missing or broken.

The batch build keeps going after an individual failure, so "the script finished" says
nothing about whether every experiment worked. This script checks, for each config:

  * a run folder exists with results_<target>.json, predictions_<target>.csv,
    videos_attempted.csv, video_features.csv and run_meta.json;
  * no classifier in results_<target>.json carries an "error";
  * all runs in a group that must be comparable (the eight factorial cells) attempted the
    same set of videos and used the same fold structure (same number of videos attempted).

    python source/scripts/verify_outputs.py                      # the eight factorial cells
    python source/scripts/verify_outputs.py --configs group_a_classical group_e_gt_upper_bound

Exit status is 1 when any check fails; the report is also written to
experiments/output_verification.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.config import load_config  # noqa: E402
from activity_detection.paths import EXPERIMENTS_DIR  # noqa: E402

FACTORIAL_CONFIGS = [
    "group_a_classical", "ablation_1_detector_only", "ablation_2_preprocessing_only",
    "ablation_3_features_only", "ablation_4_full_pipeline_fixed_mog2",
    "ablation_5_detector_preprocessing", "ablation_6_detector_features", "group_i_adaptive_mog2",
]
REQUIRED_FILES = ("videos_attempted.csv", "video_features.csv", "run_meta.json")


def latest_run(experiment_name: str, experiments_dir: Path, target: str) -> Optional[Path]:
    """Latest run folder of an experiment that produced results for the target."""
    runs = sorted(d for d in (experiments_dir / experiment_name).glob("run_*")
                  if (d / f"results_{target}.json").exists())
    return runs[-1] if runs else None


def check_run(run_dir: Path, target: str) -> List[str]:
    """Problems found in one run folder (empty list = healthy)."""
    problems = []
    for name in (f"results_{target}.json", f"predictions_{target}.csv", *REQUIRED_FILES):
        if not (run_dir / name).exists():
            problems.append(f"missing {name}")
    results_file = run_dir / f"results_{target}.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
        if not results:
            problems.append("results file is empty (training was skipped?)")
        for model, result in results.items():
            if isinstance(result, dict) and "error" in result:
                problems.append(f"model {model} failed: {str(result['error'])[:120]}")
    return problems


def attempted_videos(run_dir: Path) -> set:
    path = run_dir / "videos_attempted.csv"
    return set(pd.read_csv(path)["video"]) if path.exists() else set()


def verify(configs: List[str], targets: List[str], experiments_dir: Path,
           same_universe: List[str]) -> Dict:
    report: Dict = {"runs": {}, "problems": []}
    universes: Dict[str, set] = {}
    for config_name in configs:
        experiment_name = load_config(config_name).experiment.name
        for target in targets:
            key = f"{config_name}/{target}"
            run_dir = latest_run(experiment_name, experiments_dir, target)
            if run_dir is None:
                report["runs"][key] = {"run": None, "problems": ["no completed run found"]}
                report["problems"].append(f"{key}: no completed run found")
                continue
            problems = check_run(run_dir, target)
            meta_file = run_dir / "run_meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            report["runs"][key] = {"run": str(run_dir), "n_videos_with_features": meta.get("n_videos"),
                                   "feature_count": meta.get("feature_count"),
                                   "git_commit": meta.get("git_commit"), "problems": problems}
            report["problems"] += [f"{key}: {p}" for p in problems]
            if config_name in same_universe:
                universes[key] = attempted_videos(run_dir)

    for target in targets:
        cells = {k: v for k, v in universes.items() if k.endswith(f"/{target}")}
        if len({frozenset(v) for v in cells.values()}) > 1:
            sizes = {k.split("/")[0]: len(v) for k, v in cells.items()}
            report["problems"].append(
                f"factorial cells attempted different video sets for target '{target}': {sizes}")
    commits = {r["git_commit"] for r in report["runs"].values() if r.get("git_commit")}
    if len(commits) > 1:
        report["problems"].append(f"runs were made with different git commits: {sorted(commits)}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=FACTORIAL_CONFIGS)
    parser.add_argument("--targets", nargs="+", default=["activity", "distortion"])
    parser.add_argument("--experiments-dir", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--same-universe", nargs="*", default=FACTORIAL_CONFIGS,
                        help="configs that must have attempted exactly the same videos")
    args = parser.parse_args(argv)

    experiments_dir = Path(args.experiments_dir)
    report = verify(args.configs, args.targets, experiments_dir, args.same_universe)
    for key, info in report["runs"].items():
        status = "OK  " if not info["problems"] else "FAIL"
        print(f"  {status} {key:55s} videos={info.get('n_videos_with_features')} "
              f"features={info.get('feature_count')}")
    (experiments_dir / "output_verification.json").parent.mkdir(parents=True, exist_ok=True)
    (experiments_dir / "output_verification.json").write_text(json.dumps(report, indent=2))
    if report["problems"]:
        print(f"\n{len(report['problems'])} problem(s):")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    print("\nAll required outputs are present and error-free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
