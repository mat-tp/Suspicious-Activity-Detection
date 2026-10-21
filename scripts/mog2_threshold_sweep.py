"""Ablation on the MOG2 ``var_threshold`` (the foreground/background cut-off).

base.yaml used 25 while the literature and OpenCV value is 16 (Zivkovic and van der Heijden, 2006), and
neither was justified on this dataset. Adaptive MOG2 might also win only because it uses a lower
threshold on hard frames, so by default both the fixed detector (group_a_classical) and the adaptive
one (ablation_1_detector_only) are swept over the same grid. For the adaptive detector the threshold is
the top of its range, so values at or below the relaxed threshold are skipped.

For each base config it writes a CSV with the macro-F1 and bootstrap interval, the video yield and the
adaptive diagnostics, plus ``macro_f1_common``: the macro-F1 on the videos that survived every
threshold, because the yield changes with the threshold.

    python scripts/mog2_threshold_sweep.py --dry-run
    python scripts/mog2_threshold_sweep.py --sample --n 10
    python scripts/mog2_threshold_sweep.py --target activity --resume
    python scripts/mog2_threshold_sweep.py --values 12 16 25 --base-configs group_a_classical
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
from sklearn.metrics import f1_score

import bootstrap  # noqa: F401  (puts source/ on the import path)
from activity_detection.config import load_config
from activity_detection.experiment.runner import ExperimentRunner
from activity_detection.paths import EXPERIMENTS_DIR
from sweep_common import read_rows, run_setting

matplotlib.use("Agg")  # the plots go to files, so no window is needed
import matplotlib.pyplot as plt  # noqa: E402

# These are the variance thresholds that the sweep tries.
DEFAULT_VALUES = [8, 10, 12, 16, 20, 25, 32, 40]
# This is the threshold recommended by Zivkovic and van der Heijden (2006).
LITERATURE_VALUE = 16
# This is the threshold that every earlier result in the project used.
PREVIOUS_VALUE = 25
# These are the columns of the sweep results table.
FIELDS = ["value", "is_literature", "macro_f1", "macro_f1_low", "macro_f1_high", "accuracy",
          "balanced_accuracy", "video_yield", "trigger_rate", "accept_rate",
          "macro_f1_common", "n_common", "run_dir"]


def yield_columns(rows):
    """Names of the per-distortion yield columns present in ``rows`` (e.g. yield_Focus)."""
    return sorted({k for r in rows for k in r if k.startswith("yield_")})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-configs", nargs="+", default=["group_a_classical", "ablation_1_detector_only"])
    p.add_argument("--values", nargs="+", type=float, default=DEFAULT_VALUES)
    p.add_argument("--target", default="activity", choices=["activity", "distortion"])
    p.add_argument("--model", default="random_forest_sklearn",
                   help="the one classifier used to score every setting")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--n", type=int, default=5, help="videos per class when --sample is set")
    p.add_argument("--workers", type=int, default=0, help="video-processing worker processes")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _fmt(v: float):
    return int(v) if float(v).is_integer() else v


def make_plan(base_cfg, values):
    """(runnable values, skipped values). Adaptive needs value > tau_r to do anything."""
    if base_cfg.detection.type == "adaptive_mog2":
        tau_r = base_cfg.detection.adaptive_mog2.get("relaxed_var_threshold", 10.0)
        return [v for v in values if v > tau_r], [v for v in values if v <= tau_r]
    if base_cfg.detection.type == "mog2":
        return list(values), []
    raise SystemExit(f"{base_cfg.experiment.name} uses the {base_cfg.detection.type!r} detector; "
                     "this sweep needs mog2 or adaptive_mog2.")


def setting_config(base_name, value, model, n_videos, workers=0):
    overrides = {
        "models": {"active": [{"name": model, "params": {}}]},
        "dataset": {"n_random_videos_per_group": n_videos},
        "detection": {"mog2": {"var_threshold": _fmt(value)}},
    }
    if workers:
        overrides["experiment"] = {"n_workers": workers}
    return load_config(base_name, overrides=overrides)


def add_common_subset_scores(rows, target, model):
    """Re-score every run on the videos present in all runs, using the out-of-fold predictions.

    It warns when a predictions file or the model's column is missing, so an empty ``macro_f1_common`` is not
    mistaken for a run that did not happen.
    """
    preds = []
    for r in rows:
        f = Path(r["run_dir"]) / f"predictions_{target}.csv"
        if not f.exists():
            warnings.warn(f"macro_f1_common NOT computed: {f} does not exist.", stacklevel=2)
            return rows
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            if model not in (reader.fieldnames or []):
                warnings.warn(f"macro_f1_common NOT computed: column {model!r} not in {f} "
                              f"(columns: {reader.fieldnames}). Pass --model with one of them.",
                              stacklevel=2)
                return rows
            preds.append({row["video"]: (row["true"], row[model]) for row in reader})
    common = set.intersection(*(set(p) for p in preds)) if preds else set()
    for r, p in zip(rows, preds):
        if common:
            ys = [p[v] for v in sorted(common)]
            r["macro_f1_common"] = f1_score([t for t, _ in ys], [q for _, q in ys],
                                            average="macro", zero_division=0)
        r["n_common"] = len(common)
    return rows


def write_rows(path, rows):
    fields = FIELDS[:-1] + yield_columns(rows) + FIELDS[-1:]       # run_dir stays last
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _num(r, key):
    """float or None (CSV cells are strings and may be empty)."""
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return None


def plot(results, path):
    """Draw the macro-F1 (with and without the common videos), the video yield and the yield per distortion.
    """
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    styles = ["-", "--", ":", "-."]
    dist_names = sorted({c[len("yield_"):] for rows in results.values() for c in yield_columns(rows)})
    cmap = plt.get_cmap("tab10")
    for k, (name, rows) in enumerate(results.items()):
        rows = sorted(rows, key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in rows]
        y = [float(r["macro_f1"]) for r in rows]
        err = [[yi - float(r["macro_f1_low"]) for yi, r in zip(y, rows)],
               [float(r["macro_f1_high"]) - yi for yi, r in zip(y, rows)]]
        line = a1.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{name} (all videos)")
        common = [(xi, _num(r, "macro_f1_common")) for xi, r in zip(x, rows)]
        common = [(xi, c) for xi, c in common if c is not None]
        if common:
            a1.plot(*zip(*common), ls="--", marker="s", ms=4, color=line[0].get_color(),
                    label=f"{name} (common subset)")
        a2.plot(x, [float(r["video_yield"]) for r in rows], marker="o", label=name)
        for j, d in enumerate(dist_names):
            pts = [(xi, _num(r, f"yield_{d}")) for xi, r in zip(x, rows)]
            pts = [(xi, v) for xi, v in pts if v is not None]
            if pts:
                a3.plot(*zip(*pts), ls=styles[k % len(styles)], marker="o", ms=3,
                        color=cmap(j % 10), label=f"{d} | {name}")
    for ax in (a1, a2, a3):
        ax.axvline(LITERATURE_VALUE, color="green", ls="--", lw=1)
        ax.axvline(PREVIOUS_VALUE, color="grey", ls=":", lw=1)
        ax.set_xlabel("MOG2 var_threshold")
        ax.grid(alpha=0.3)
    a1.set_ylabel("macro-F1 (95% bootstrap CI)")
    a2.set_ylabel("video yield")
    a3.set_ylabel("video yield per distortion")
    a1.legend(fontsize=7)
    a2.legend(fontsize=8)
    if a3.get_legend_handles_labels()[0]:
        a3.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def sweep(args, runner_class=None):
    plans = {}
    for name in args.base_configs:
        cfg = load_config(name)
        runnable, skipped = make_plan(cfg, args.values)
        plans[name] = runnable
        print(f"{name} ({cfg.detection.type}): run {[_fmt(v) for v in runnable]}"
              + (f"  | skipped (<= tau_r, adaptation would be a no-op): {[_fmt(v) for v in skipped]}" if skipped else ""))
    print(f"target={args.target} model={args.model}; {sum(map(len, plans.values()))} full pipeline runs.")
    if args.dry_run:
        return {}
    runner_class = runner_class or ExperimentRunner

    results = {}
    for name, values in plans.items():
        out_dir = EXPERIMENTS_DIR / f"mog2_threshold_{name}_{args.target}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "threshold_results.csv"
        rows = read_rows(csv_path) if args.resume else []
        done = {float(r["value"]) for r in rows}
        for v in values:
            if v in done:
                print(f"Skipping {name} var_threshold={_fmt(v)} (already done)")
                continue
            print(f"\n{'=' * 70}\n  {name}  var_threshold = {_fmt(v)}\n{'=' * 70}")
            cfg = setting_config(name, v, args.model, args.n, args.workers)
            row = run_setting(cfg, f"mog2thr_{name}_{_fmt(v)}", args.target, args.sample, runner_class)
            rows.append({"value": _fmt(v), "is_literature": v == LITERATURE_VALUE, **row})
            write_rows(csv_path, rows)                 # saved after every setting
        rows = add_common_subset_scores(rows, args.target, args.model)
        write_rows(csv_path, rows)
        results[name] = rows
        print(f"Results: {csv_path}")
    if results:
        plot(results, EXPERIMENTS_DIR / f"mog2_threshold_{args.target}.png")
    return results


if __name__ == "__main__":
    sweep(parse_args())
