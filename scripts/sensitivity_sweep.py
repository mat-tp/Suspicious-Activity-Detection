"""Sensitivity of adaptive MOG2 to its four hyperparameters.

Varies one parameter at a time, keeping the others at the config's values:

    q        quality_threshold        adapt when the combined score is below q
    w        mask_weight              weight of mask quality vs track quality
    tau_r    relaxed_var_threshold    threshold under full relaxation
    tau_min  threshold_floor          the threshold never goes below this

Every setting runs the full pipeline (detection, tracking, features, cross-
validation), so a full sweep takes hours. Results are written after each
setting, and --resume skips settings that are already in the CSV.

Examples (run from the project root):

    python source/scripts/sensitivity_sweep.py --dry-run
    python source/scripts/sensitivity_sweep.py --sample --n 10          # quick check
    python source/scripts/sensitivity_sweep.py --params q w --target activity
    python source/scripts/sensitivity_sweep.py --values q=0.3,0.4,0.5 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.config import load_config  # noqa: E402
from activity_detection.paths import EXPERIMENTS_DIR  # noqa: E402

# name used in the paper -> (config key, values to try)
PARAMETERS = {
    "q":       ("quality_threshold",      [0.2, 0.3, 0.4, 0.5, 0.6]),
    "w":       ("mask_weight",            [0.5, 0.6, 0.7, 0.8, 0.9]),
    "tau_r":   ("relaxed_var_threshold",  [8, 10, 12, 15, 18]),
    "tau_min": ("threshold_floor",        [5, 8, 10, 12, 15]),
}
FIELDS = ["param", "value", "is_default", "macro_f1", "macro_f1_low", "macro_f1_high",
          "accuracy", "balanced_accuracy", "video_yield", "trigger_rate", "accept_rate", "run_dir"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", default="ablation_1_detector_only",
                   help="config to sweep (must use the adaptive_mog2 detector)")
    p.add_argument("--params", nargs="+", choices=list(PARAMETERS), default=list(PARAMETERS))
    p.add_argument("--values", nargs="+", default=[], metavar="P=v1,v2,...",
                   help="replace the default values of a parameter, e.g. q=0.3,0.5")
    p.add_argument("--target", default="activity", choices=["activity", "distortion"])
    p.add_argument("--model", default="random_forest_sklearn",
                   help="the one classifier to score each setting with (fast and stable by default)")
    p.add_argument("--sample", action="store_true", help="use a small subset of videos")
    p.add_argument("--n", type=int, default=5, help="videos per class when --sample is set")
    p.add_argument("--resume", action="store_true", help="skip settings already in the CSV")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def build_grid(selected, overrides):
    """{param: [values]} with any --values overrides applied."""
    grid = {name: list(PARAMETERS[name][1]) for name in selected}
    for item in overrides:
        name, _, text = item.partition("=")
        if name not in PARAMETERS:
            raise SystemExit(f"Unknown parameter in --values: {name!r}. Use one of {list(PARAMETERS)}")
        grid[name] = [float(v) for v in text.split(",") if v]
    return {k: v for k, v in grid.items() if k in selected}


def make_plan(grid, default_values):
    """List of (param, value). The default setting is run once and shared."""
    plan = [("default", None)]
    for name, values in grid.items():
        default = default_values[PARAMETERS[name][0]]
        plan += [(name, v) for v in values if v != default]
    return plan


def setting_config(base, param, value, model, n_videos):
    overrides = {
        "models": {"active": [{"name": model, "params": {}}]},
        "dataset": {"n_random_videos_per_group": n_videos},
    }
    if param != "default":
        overrides["detection"] = {"adaptive_mog2": {PARAMETERS[param][0]: value}}
    return load_config(base, overrides=overrides)


def run_setting(cfg, name, target, use_sample, runner_class):
    """Run one setting and return the numbers for one CSV row."""
    runner = runner_class(cfg, experiment_name=name)
    df = runner.load_dataset()
    n_total = len(df)
    features = runner.collect_video_features(df, use_sample=use_sample)
    results = runner.train_and_evaluate(features, target_col=target)

    model_result = next((r for r in results.values() if "cv" in r), None)
    if model_result is None:
        raise RuntimeError(f"No model finished for {name}: {results}")
    cv = model_result["cv"]
    row = {
        "macro_f1": cv["pooled"]["macro_f1"],
        "macro_f1_low": cv["ci95"]["macro_f1"][0], "macro_f1_high": cv["ci95"]["macro_f1"][1],
        "accuracy": cv["pooled"]["accuracy"],
        "balanced_accuracy": cv["pooled"]["balanced_accuracy"],
        "video_yield": len(features) / max(n_total, 1),
        "trigger_rate": "", "accept_rate": "", "run_dir": str(runner.run_dir),
    }
    diag_file = Path(runner.run_dir) / "adaptive_mog2_diagnostics.json"
    if diag_file.exists():
        diag = json.loads(diag_file.read_text())
        if diag:
            row["trigger_rate"] = sum(d["adaptive_trigger_rate"] for d in diag) / len(diag)
            row["accept_rate"] = sum(d["rerun_accept_rate"] for d in diag) / len(diag)
    return row


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, grid, default_values, path):
    """One panel per parameter: macro-F1 with its 95% interval."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the plot.")
        return
    default_row = next((r for r in rows if r["param"] == "default"), None)
    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 3.6), squeeze=False, sharey=True)
    for ax, (name, _) in zip(axes[0], grid.items()):
        points = [r for r in rows if r["param"] == name]
        if default_row:
            points.append({**default_row, "value": default_values[PARAMETERS[name][0]]})
        points.sort(key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in points]
        y = [float(r["macro_f1"]) for r in points]
        err = [[y_i - float(r["macro_f1_low"]) for y_i, r in zip(y, points)],
               [float(r["macro_f1_high"]) - y_i for y_i, r in zip(y, points)]]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3)
        ax.axvline(default_values[PARAMETERS[name][0]], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("macro-F1 (95% bootstrap interval)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def sweep(args, runner_class=None):
    base = load_config(args.base_config)
    if base.detection.type != "adaptive_mog2":
        raise SystemExit(f"{args.base_config} uses the {base.detection.type!r} detector; "
                         "the sweep needs adaptive_mog2.")
    defaults = base.detection.adaptive_mog2.to_dict()
    grid = build_grid(args.params, args.values)
    plan = make_plan(grid, defaults)

    print(f"Sweeping {args.base_config}  (target={args.target}, model={args.model})")
    print("Defaults: " + ", ".join(f"{n}={defaults[PARAMETERS[n][0]]}" for n in PARAMETERS))
    for param, value in plan:
        print(f"  {param:8s} {'(config values)' if value is None else value}")
    print(f"{len(plan)} settings, each a full pipeline run.")
    if args.dry_run:
        return []

    if runner_class is None:
        from activity_detection.experiment.runner import (
            ExperimentRunner as runner_class,
        )
    out_dir = EXPERIMENTS_DIR / f"sensitivity_{args.base_config}_{args.target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sensitivity_results.csv"
    rows = read_rows(csv_path) if args.resume else []
    done = {(r["param"], r["value"]) for r in rows}

    for param, value in plan:
        key = (param, "" if value is None else str(value))
        if key in done:
            print(f"Skipping {param}={value} (already done)")
            continue
        print(f"\n{'=' * 70}\n  {param} = {value if value is not None else 'config values'}\n{'=' * 70}")
        cfg = setting_config(args.base_config, param, value, args.model, args.n)
        name = f"sensitivity_{args.base_config}_{param}_{value}"
        row = run_setting(cfg, name, args.target, args.sample, runner_class)
        rows.append({"param": param, "value": "" if value is None else value,
                     "is_default": param == "default", **row})
        write_rows(csv_path, rows)          # saved after every setting

    plot(rows, grid, defaults, out_dir / "sensitivity.png")
    print(f"\nResults: {csv_path}")
    return rows


if __name__ == "__main__":
    sweep(parse_args())
