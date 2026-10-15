"""Command-line interface."""
from __future__ import annotations

import argparse

from . import registry
from .config import load_config, merge_overrides, parse_set_overrides
from .experiment.runner import ExperimentRunner
from .optional import TORCH_AVAILABLE
from .paths import CONFIGS_DIR
from .pipeline.builder import DetectionSource

COMPONENT_KINDS = ["detector", "tracker", "preprocessor", "feature", "model"]

EXAMPLES = """
examples:
  python source/run.py --config group_a_classical --target activity
  python source/run.py --config group_e_gt_upper_bound --detection-source ground_truth
  python source/run.py --config group_a_classical --diagnose
  python source/run.py --all-groups --target both
  python source/run.py --ablations --target activity
  python source/run.py --detectors --target activity
  python source/run.py --list-configs
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Activity and imaging-condition classification on AD-SVD.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", "-c", help="name of a config in source/configs (without .yaml)")
    p.add_argument("--all-groups", action="store_true", help="run every group_*.yaml config")
    p.add_argument("--ablations", action="store_true", help="run every ablation_*.yaml config")
    p.add_argument("--detectors", action="store_true",
                   help="run every detector_*.yaml config (ViBe, PBAS, SuBSENSE vs MOG2)")
    p.add_argument("--target", "-t", default="activity", choices=["activity", "distortion", "both"],
                   help="what to classify (default: activity)")
    p.add_argument("--sample", action="store_true", help="use a small random subset of videos")
    p.add_argument("--n", type=int, default=5, help="videos per class when --sample is set")
    p.add_argument("--detection-source", "-ds", choices=["mog2", "ground_truth"],
                   help="override the detection source from the config")
    p.add_argument("--diagnose", action="store_true", help="extract features only, do not train")
    p.add_argument("--visualise", action="store_true", help="write FiftyOne visualisations")
    p.add_argument("--n-visualise", type=int, default=3, help="visualised videos per activity class")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one config value, e.g. --set detection.mog2.history=300 "
                        "(repeatable; values are parsed as YAML)")
    p.add_argument("--workers", type=int, help="number of video-processing worker processes "
                                               "(overrides experiment.n_workers)")
    p.add_argument("--features-csv", help="reuse a cached video_features.csv instead of "
                                          "processing videos (classifier-only re-runs)")
    p.add_argument("--list-configs", action="store_true", help="list available configs and exit")
    p.add_argument("--list-components", action="store_true", help="list registered components and exit")
    return p


def list_configs() -> None:
    print("Available configs:")
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        if path.stem != "base":
            print(f"  {path.stem}")


def list_components() -> None:
    for kind in COMPONENT_KINDS:
        print(f"\n{kind}:")
        for name in registry.available(kind):
            print(f"  {name}")


def run_group(group: str, args: argparse.Namespace) -> None:
    print(f"\n{'#' * 70}\n# GROUP: {group}\n{'#' * 70}")
    try:
        overrides = merge_overrides(
            {"dataset": {"n_random_videos_per_group": args.n}},
            {"experiment": {"n_workers": args.workers}} if args.workers else None,
            parse_set_overrides(args.overrides),
        )
        cfg = load_config(group, overrides=overrides)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        return

    data_path = cfg.get("data_path", "tabular")
    if data_path in ("clip", "hybrid") and not TORCH_AVAILABLE:
        print(f"  SKIPPING {group}: it needs PyTorch, which is not installed.")
        return

    source = DetectionSource(args.detection_source) if args.detection_source else None
    runner = ExperimentRunner(cfg, detection_source=source, diagnose_only=args.diagnose,
                              visualise=args.visualise, n_visualise=args.n_visualise)
    df = runner.load_dataset()

    targets = ["activity", "distortion"] if args.target == "both" else [args.target]
    try:
        # Video processing takes hours, so tabular runs collect features once and reuse
        # them for every target.
        cached_features = None
        if data_path != "clip":
            cached_features = (ExperimentRunner.load_video_features(args.features_csv)
                               if args.features_csv else runner.collect_video_features(df, use_sample=args.sample))
        for target in targets:
            print(f"\n  --- Target: {target} ---")
            if data_path == "clip":
                run_neural(runner, df, target, args)
            else:
                run_tabular(runner, cached_features, target, args)
    except ImportError as err:
        print(f"  SKIPPING {group}: {err}")


def run_neural(runner, df, target, args) -> None:
    clips = runner.prepare_clip_dataset(df, use_sample=args.sample)
    if not clips:
        print("  No clips generated.")
    elif args.diagnose:
        print("  Diagnose mode: skipping neural training.")
    else:
        runner.train_neural_model(clips, target_col=target)


def run_tabular(runner, features, target, args) -> None:
    if not features:
        print(f"  No video features collected for target '{target}'.")
    elif args.diagnose:
        print(f"  Diagnose mode: collected {len(features)} feature vectors, not training.")
    else:
        runner.train_and_evaluate(features, target_col=target)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return 0
    if args.list_components:
        list_components()
        return 0

    if args.all_groups:
        groups = sorted(p.stem for p in CONFIGS_DIR.glob("group_*.yaml"))
    elif args.detectors:
        groups = sorted(p.stem for p in CONFIGS_DIR.glob("detector_*.yaml"))
    elif args.ablations:
        groups = sorted(p.stem for p in CONFIGS_DIR.glob("ablation_*.yaml"))
    elif args.config:
        groups = [args.config]
    else:
        parser.print_help()
        return 1

    try:
        for group in groups:
            run_group(group, args)
    except FileNotFoundError as err:
        print(f"\nERROR: {err}\nSee docs/howToRun.md, step 2, for where the dataset goes.")
        return 1

    print("\nDone. Results are in experiments/")
    return 0
