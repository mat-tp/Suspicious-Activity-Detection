"""One-at-a-time ablation over the selected parameters of the pipeline.

The reference is the full adaptive pipeline (``group_i_adaptive_mog2``). Each parameter is varied over a
list of values while all others stay at the reference, so a change in macro-F1 belongs to that parameter.

A ``pipeline`` parameter changes detection, tracking or features, so the videos are reprocessed for every
value (hours per setting). A ``model`` parameter changes only the classifier or the imbalance handling
and reuses the cached features (minutes). Both targets come from one video pass, results are appended
to a CSV after every setting, and ``--resume`` skips finished settings.

    python scripts/parameter_ablation.py --list
    python scripts/parameter_ablation.py --dry-run
    python scripts/parameter_ablation.py --sample --n 5 --groups detector
    python scripts/parameter_ablation.py --resume
    python scripts/parameter_ablation.py --groups classifier imbalance
"""
from __future__ import annotations

import argparse
import copy
import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

import bootstrap  # noqa: F401  (puts source/ on the import path)
from activity_detection.config import load_config, merge_overrides
from activity_detection.experiment.runner import ExperimentRunner
from activity_detection.paths import EXPERIMENTS_DIR

matplotlib.use("Agg")  # the plots go to files, so no window is needed
import matplotlib.pyplot as plt  # noqa: E402

# This is the config whose other settings stay fixed while one parameter changes.
REFERENCE_CONFIG = "group_i_adaptive_mog2"
# This is the classifier used to score each parameter value.
DEFAULT_SCORING_MODEL = "random_forest_sklearn"

# These are the columns of the results table.
RESULT_FIELDS = ["group", "parameter", "value", "is_reference", "stage", "model", "target",
                 "macro_f1", "macro_f1_low", "macro_f1_high", "accuracy", "balanced_accuracy",
                 "n_videos", "n_features", "run_dir"]


@dataclass
class Parameter:
    """One ablated parameter.

    Attributes:
        group: the group of parameters it belongs to.
        name: the name shown in the results.
        config_path: dotted path into the config (pipeline) or into the model's ``params`` (model).
        values: the values to try; the reference value runs once and is shared.
        stage: ``"pipeline"`` (reprocess the videos) or ``"model"`` (reuse the cached features).
        model: for the model stage, the registry name of the classifier being tuned.
        companion: extra overrides for every setting of this parameter, including its reference run.
    """

    group: str
    name: str
    config_path: str
    values: List[Any]
    stage: str = "pipeline"
    model: Optional[str] = None
    companion: Dict[str, Any] = field(default_factory=dict)


def _pipeline(group, name, path, values, **kw):
    return Parameter(group, name, path, values, "pipeline", **kw)


def _model(group, model, name, values):
    return Parameter(group, f"{model}.{name}", name, values, "model", model=model)


# Every parameter set in configs/base.yaml and configs/group_i_adaptive_mog2.yaml.
PARAMETER_TABLE: List[Parameter] = [
    _pipeline("preprocessing", "frame_resize", "preprocessing.frame_resize",
              [[320, 240], [480, 360], [640, 480], [800, 600]]),
    _pipeline("preprocessing", "blur_kernel", "preprocessing.blur_kernel",
              [[3, 3], [5, 5], [7, 7]]),
    _pipeline("preprocessing", "blur_threshold",
              "preprocessing.distortion_aware_params.blur_threshold", [50.0, 100.0, 150.0, 200.0]),
    _pipeline("preprocessing", "exposure_low",
              "preprocessing.distortion_aware_params.exposure_low", [40.0, 60.0, 80.0]),
    _pipeline("preprocessing", "exposure_high",
              "preprocessing.distortion_aware_params.exposure_high", [175.0, 195.0, 215.0]),
    _pipeline("preprocessing", "unsharp_sigma",
              "preprocessing.distortion_aware_params.unsharp_sigma", [1.0, 2.0, 3.0]),
    _pipeline("preprocessing", "unsharp_strength",
              "preprocessing.distortion_aware_params.unsharp_strength", [0.5, 1.0, 1.5, 2.0]),
    _pipeline("preprocessing", "clahe_clip_limit",
              "preprocessing.distortion_aware_params.clahe_clip_limit", [1.0, 2.0, 3.0, 4.0]),
    _pipeline("preprocessing", "clahe_tile_grid",
              "preprocessing.distortion_aware_params.clahe_tile_grid", [[4, 4], [8, 8], [16, 16]]),

    _pipeline("detector", "mog2_history", "detection.mog2.history", [10, 20, 50, 100]),
    _pipeline("detector", "mog2_var_threshold", "detection.mog2.var_threshold",
              [9.0, 12.0, 16.0, 20.0, 25.0, 36.0]),
    _pipeline("detector", "min_contour_area", "detection.min_contour_area", [250, 500, 1000, 2000]),
    _pipeline("detector", "morph_close_iter", "detection.morphology.close_iter", [1, 2, 3]),
    _pipeline("detector", "morph_open_iter", "detection.morphology.open_iter", [0, 1, 2]),

    _pipeline("adaptive", "q_quality_threshold", "detection.adaptive_mog2.quality_threshold",
              [0.2, 0.3, 0.4, 0.5, 0.6]),
    _pipeline("adaptive", "w_mask_weight", "detection.adaptive_mog2.mask_weight",
              [0.5, 0.6, 0.7, 0.8, 0.9]),
    _pipeline("adaptive", "tau_r_relaxed_var_threshold",
              "detection.adaptive_mog2.relaxed_var_threshold", [8.0, 10.0, 12.0, 15.0]),
    _pipeline("adaptive", "tau_min_threshold_floor",
              "detection.adaptive_mog2.threshold_floor", [5.0, 8.0, 10.0, 12.0]),
    _pipeline("adaptive", "ema_alpha", "detection.adaptive_mog2.ema_alpha", [0.5, 0.8, 0.95]),
    _pipeline("adaptive", "adaptive_min_track_len", "detection.adaptive_mog2.min_track_len",
              [3, 5, 10]),
    _pipeline("adaptive", "update_once", "detection.adaptive_mog2.update_once", [True, False]),

    _pipeline("tracker", "iou_threshold", "tracking.iou_hungarian.iou_threshold",
              [0.1, 0.2, 0.3, 0.4, 0.5]),
    _pipeline("tracker", "max_disappeared", "tracking.max_disappeared", [10, 20, 30, 50]),
    _pipeline("tracker", "max_trajectory_len", "tracking.max_trajectory_len", [150, 300, 600]),
    # Filtering must be on for the minimum length to change anything, so it is a companion
    # of every setting (the reference setting 15 is re-run with filtering on).
    _pipeline("tracker", "min_trajectory_len_filtered", "tracking.min_trajectory_len",
              [5, 15, 30, 45], companion={"tracking.filter_short_tracks": True}),

    _pipeline("features", "aggregation", "features.aggregation",
              [["mean"], ["mean", "max"], ["mean", "max", "std"]]),
    _pipeline("features", "wavelet_level", "features.module_params.wavelet.level", [2, 3, 4]),
    _pipeline("features", "wavelet_family", "features.module_params.wavelet.wavelet",
              ["db2", "db4", "sym4"]),
    _pipeline("features", "interaction_proximity_threshold",
              "features.module_params.interaction.proximity_threshold", [50.0, 100.0, 150.0, 200.0]),
    _pipeline("features", "global_flow_step", "features.global_flow_step", [2, 5, 10]),
    _pipeline("features", "appearance_frame_count", "features.appearance_frame_count", [10, 30, 60]),
    _pipeline("features", "lk_max_corners", "optical_flow.max_corners", [100, 200, 300]),
    _pipeline("features", "lk_win_size", "optical_flow.lk_win_size", [[9, 9], [15, 15], [21, 21]]),
    _pipeline("features", "lk_corner_quality", "optical_flow.corner_quality", [0.1, 0.3, 0.5]),

    _model("classifier", "random_forest_sklearn", "n_estimators", [50, 100, 200, 400]),
    _model("classifier", "random_forest_sklearn", "max_depth", [5, 10, 20, None]),
    _model("classifier", "svm_rbf", "C", [0.1, 1, 10, 100]),
    _model("classifier", "svm_rbf", "gamma", ["scale", "auto", 0.001, 0.01]),
    _model("classifier", "knn", "n_neighbors", [3, 5, 9, 15]),
    _model("classifier", "knn", "metric", ["euclidean", "manhattan", "cosine"]),
    _model("classifier", "random_forest_custom", "n_trees", [50, 100, 200]),
    _model("classifier", "random_forest_custom", "max_depth", [5, 10, 20]),
    _model("classifier", "lightgbm", "n_estimators", [100, 200, 400]),
    _model("classifier", "lightgbm", "max_depth", [3, 6, 9]),
    _model("classifier", "lightgbm", "learning_rate", [0.03, 0.1, 0.3]),
    _model("classifier", "lightgbm", "num_leaves", [15, 31, 63]),
    _model("classifier", "xgboost", "n_estimators", [100, 200, 400]),
    _model("classifier", "xgboost", "max_depth", [3, 6, 9]),
    _model("classifier", "xgboost", "learning_rate", [0.03, 0.1, 0.3]),
    _model("classifier", "mlp", "hidden_layer_sizes", [[64], [128, 64], [256, 128]]),
    _model("classifier", "mlp", "alpha", [0.0001, 0.001, 0.01]),

    Parameter("imbalance", "resampling", "imbalance.resampling",
              ["none", "random_over", "smote", "borderline_smote"], stage="model",
              model=DEFAULT_SCORING_MODEL),
    Parameter("imbalance", "class_weight", "imbalance.class_weight", ["balanced", "none"],
              stage="model", model=DEFAULT_SCORING_MODEL),
]

# These are the names of the parameter groups, in alphabetical order.
GROUP_NAMES = sorted({p.group for p in PARAMETER_TABLE})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-config", default=REFERENCE_CONFIG,
                        help="the config every setting is derived from")
    parser.add_argument("--groups", nargs="+", choices=GROUP_NAMES, default=GROUP_NAMES,
                        help="parameter groups to run (default: all)")
    parser.add_argument("--only", nargs="+", default=[],
                        help="restrict to these parameter names (see --list)")
    parser.add_argument("--targets", nargs="+", default=["activity", "distortion"],
                        choices=["activity", "distortion"])
    parser.add_argument("--scoring-model", default=DEFAULT_SCORING_MODEL,
                        help="classifier that scores pipeline-stage settings")
    parser.add_argument("--sample", action="store_true", help="use a small subset of videos")
    parser.add_argument("--n", type=int, default=5, help="videos per class when --sample is set")
    parser.add_argument("--workers", type=int, default=0, help="video-processing workers")
    parser.add_argument("--resume", action="store_true", help="skip settings already in the CSV")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--list", action="store_true", help="list parameters and exit")
    parser.add_argument("--output-name", default="parameter_ablation",
                        help="folder name under experiments/")
    return parser.parse_args(argv)


def get_by_path(mapping: dict, dotted_path: str, default=None):
    node = mapping
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_by_path(mapping: dict, dotted_path: str, value) -> dict:
    """Return a nested override dict {a: {b: {c: value}}} for 'a.b.c'."""
    *parents, leaf = dotted_path.split(".")
    nested: dict = {leaf: value}
    for key in reversed(parents):
        nested = {key: nested}
    return merge_overrides(mapping, nested)


def reference_value(parameter: Parameter, reference_config: dict, scoring_model: str):
    """Value of the parameter in the reference config."""
    if parameter.stage == "pipeline" or parameter.group == "imbalance":
        return get_by_path(reference_config, parameter.config_path)
    for model_def in reference_config["models"]["active"]:
        if model_def["name"] == parameter.model:
            return (model_def.get("params") or {}).get(parameter.config_path)
    return None


def select_parameters(groups, only) -> List[Parameter]:
    chosen = [p for p in PARAMETER_TABLE if p.group in groups]
    if only:
        unknown = set(only) - {p.name for p in PARAMETER_TABLE}
        if unknown:
            raise SystemExit(f"Unknown parameter name(s) in --only: {sorted(unknown)}")
        chosen = [p for p in chosen if p.name in only]
    return chosen


def make_plan(parameters, reference_config: dict, scoring_model: str):
    """List of settings: (parameter, value, is_reference). The reference is run once."""
    plan = [(None, None, True)]
    for parameter in parameters:
        reference = reference_value(parameter, reference_config, scoring_model)
        for value in parameter.values:
            shares_reference = (value == reference) and not parameter.companion
            if not shares_reference:
                plan.append((parameter, value, value == reference))
    return plan


def setting_key(parameter, value) -> tuple:
    return ("reference", "") if parameter is None else (parameter.name, str(value))


def pipeline_overrides(parameter: Parameter, value, n_videos: int, workers: int,
                       scoring_model: str) -> dict:
    overrides = {"dataset": {"n_random_videos_per_group": n_videos},
                 "models": {"active": [{"name": scoring_model, "params": {}}]}}
    if workers:
        overrides["experiment"] = {"n_workers": workers}
    if parameter is not None:
        overrides = set_by_path(overrides, parameter.config_path, copy.deepcopy(value))
        for path, companion_value in parameter.companion.items():
            overrides = set_by_path(overrides, path, companion_value)
    return overrides


def model_overrides(parameter: Parameter, value, reference_config: dict, n_videos: int) -> dict:
    """Overrides for a model-stage setting: only the tuned model is evaluated."""
    base_params: dict = {}
    for model_def in reference_config["models"]["active"]:
        if model_def["name"] == parameter.model:
            base_params = copy.deepcopy(model_def.get("params") or {})
    overrides = {"dataset": {"n_random_videos_per_group": n_videos}}
    if parameter.group == "imbalance":
        overrides = set_by_path(overrides, parameter.config_path, value)
        overrides["models"] = {"active": [{"name": parameter.model, "params": base_params}]}
    else:
        base_params[parameter.config_path] = copy.deepcopy(value)
        overrides["models"] = {"active": [{"name": parameter.model, "params": base_params}]}
    return overrides


def rows_from_results(results: dict, parameter, value, is_reference, stage, target,
                      n_videos, run_dir) -> List[dict]:
    rows = []
    for model_name, result in results.items():
        if "cv" not in result:
            continue
        cv = result["cv"]
        rows.append({
            "group": parameter.group if parameter else "reference",
            "parameter": parameter.name if parameter else "reference",
            "value": "" if parameter is None else str(value),
            "is_reference": is_reference, "stage": stage, "model": model_name, "target": target,
            "macro_f1": cv["pooled"]["macro_f1"],
            "macro_f1_low": cv["ci95"]["macro_f1"][0], "macro_f1_high": cv["ci95"]["macro_f1"][1],
            "accuracy": cv["pooled"]["accuracy"],
            "balanced_accuracy": cv["pooled"]["balanced_accuracy"],
            "n_videos": n_videos, "n_features": result.get("feature_count", ""),
            "run_dir": str(run_dir),
        })
    return rows


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_ablation(args, runner_class=None) -> List[dict]:
    reference = load_config(args.reference_config)
    reference_dict = reference.to_dict()
    parameters = select_parameters(args.groups, args.only)
    plan = make_plan(parameters, reference_dict, args.scoring_model)

    print(f"Reference config: {args.reference_config}   scoring model: {args.scoring_model}")
    n_pipeline = sum(1 for p, _, _ in plan if p is None or p.stage == "pipeline")
    n_model = len(plan) - n_pipeline
    for parameter, value, _ in plan:
        label = "reference" if parameter is None else f"{parameter.name} = {value}"
        stage = "pipeline" if parameter is None else parameter.stage
        print(f"  [{stage:8s}] {label}")
    print(f"{len(plan)} settings: {n_pipeline} reprocess videos, {n_model} reuse cached features.")
    if args.dry_run:
        return []

    runner_class = runner_class or ExperimentRunner

    out_dir = EXPERIMENTS_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_results.csv"
    (out_dir / "failures.txt").unlink(missing_ok=True)     # a stale file must not fail this run
    cache_path = out_dir / "reference_video_features.csv"
    rows = read_rows(csv_path) if args.resume else []
    done = {(r["parameter"], r["value"]) for r in rows}
    cached_features = (runner_class.load_video_features(cache_path)
                       if args.resume and cache_path.exists() else None)

    failures: List[str] = []
    for parameter, value, is_reference in plan:
        key = setting_key(parameter, value)
        if key in done:
            print(f"Skipping {key[0]}={key[1]} (already done)")
            continue
        label = "reference" if parameter is None else f"{parameter.name}={value}"
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
        try:
            if parameter is None or parameter.stage == "pipeline":
                overrides = pipeline_overrides(parameter, value, args.n, args.workers,
                                               args.scoring_model)
                config = load_config(args.reference_config, overrides=overrides)
                runner = runner_class(config, experiment_name=f"{args.output_name}_{label}")
                dataset = runner.load_dataset()
                features = runner.collect_video_features(dataset, use_sample=args.sample)
                if parameter is None:
                    cached_features = features
                    reference_cache = Path(runner.run_dir) / "video_features.csv"
                    if reference_cache.exists():
                        shutil.copy(reference_cache, cache_path)
                stage = "pipeline"
            else:
                if cached_features is None:
                    raise RuntimeError("model-stage setting needs the reference features; "
                                       "run the reference first (do not use --only with "
                                       "model parameters on a fresh output folder)")
                overrides = model_overrides(parameter, value, reference_dict, args.n)
                config = load_config(args.reference_config, overrides=overrides)
                runner = runner_class(config, experiment_name=f"{args.output_name}_{label}")
                features = cached_features
                stage = "model"
            for target in args.targets:
                results = runner.train_and_evaluate(features, target_col=target)
                rows += rows_from_results(results, parameter, value, is_reference, stage,
                                          target, len(features), runner.run_dir)
            write_rows(csv_path, rows)              # saved after every setting
        except Exception as error:                  # keep the batch going, report at the end
            failures.append(f"{label}: {error}")
            print(f"  FAILED {label}: {error}")

    print(f"\nResults: {csv_path}")
    plot_groups(rows, out_dir)
    if failures:
        print(f"\n{len(failures)} setting(s) FAILED:")
        for line in failures:
            print(f"  - {line}")
        (out_dir / "failures.txt").write_text("\n".join(failures))
    return rows


def plot_groups(rows: List[dict], out_dir: Path) -> None:
    """One figure per parameter group and target: macro-F1 (95% interval) per tested value."""
    reference_rows = {r["target"]: r for r in rows if r["parameter"] == "reference"}
    for target in sorted({r["target"] for r in rows}):
        for group in GROUP_NAMES:
            names = [p.name for p in PARAMETER_TABLE if p.group == group
                     and any(r["parameter"] == p.name and r["target"] == target for r in rows)]
            if not names:
                continue
            columns = min(4, len(names))
            n_rows = -(-len(names) // columns)
            fig, axes = plt.subplots(n_rows, columns, figsize=(3.6 * columns, 3.0 * n_rows),
                                     squeeze=False, sharey=True)
            for ax in axes.flat[len(names):]:
                ax.axis("off")
            for ax, name in zip(axes.flat, names):
                parameter = next(p for p in PARAMETER_TABLE if p.name == name)
                points = [r for r in rows if r["parameter"] == name and r["target"] == target
                          and (parameter.group != "classifier" or r["model"] == parameter.model)]
                # The reference setting is shared, so add it explicitly where it belongs.
                reference_row = reference_rows.get(target)
                if reference_row and not any(r["is_reference"] == "True" for r in points):
                    ref_value = reference_value(parameter, load_config(REFERENCE_CONFIG).to_dict(),
                                                DEFAULT_SCORING_MODEL)
                    points.append({**reference_row, "value": str(ref_value)})
                order = {str(v): i for i, v in enumerate(parameter.values)}
                points.sort(key=lambda r: order.get(r["value"], len(order)))
                x = list(range(len(points)))
                y = [float(r["macro_f1"]) for r in points]
                err = [[yi - float(r["macro_f1_low"]) for yi, r in zip(y, points)],
                       [float(r["macro_f1_high"]) - yi for yi, r in zip(y, points)]]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3)
                ax.set_xticks(x)
                ax.set_xticklabels([r["value"] for r in points], rotation=45, ha="right", fontsize=7)
                ax.set_title(name, fontsize=8)
                ax.grid(alpha=0.3)
            fig.supylabel("macro-F1")
            fig.tight_layout()
            path = out_dir / f"ablation_{group}_{target}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Plot saved: {path}")


def print_parameter_list() -> None:
    for parameter in PARAMETER_TABLE:
        print(f"{parameter.group:14s} {parameter.stage:8s} {parameter.name:36s} {parameter.values}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.list:
        print_parameter_list()
        sys.exit(0)
    run_ablation(arguments)
    failure_file = EXPERIMENTS_DIR / arguments.output_name / "failures.txt"
    sys.exit(1 if failure_file.exists() else 0)
