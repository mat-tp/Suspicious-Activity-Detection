"""2x2x2 factorial ablation: detector x preprocessing x features.

Answers SACAIR Reviewer 3, who pointed out that the baseline / ablation / proposed
configurations changed several things at once, so the gain could not be credited
to adaptive MOG2. Here all eight combinations are run (each factor on or off) and
the effect of every factor is estimated while the others are held fixed.

    D = detector       0 fixed MOG2          1 adaptive MOG2
    P = preprocessing  0 standard            1 distortion-aware
    F = features       0 trajectory only     1 multi-family (221-dim)

Reported, each with a 95% bootstrap interval over videos (the SAME resampled
videos for all eight cells, so the cells are paired):

  * the 8 cell scores
  * simple effect of each factor at every setting of the other two
    (e.g. adaptive - fixed MOG2 with standard preproc + trajectory features, ...)
  * main effect of each factor (average of its simple effects)
  * two-way interactions (does the detector effect depend on preprocessing / features?)

Videos: ``common`` = videos every cell produced a prediction for; ``all`` = every
video attempted by every cell, with a missing prediction counted as a miss (the
end-to-end view, which includes detector yield).

    python source/scripts/factorial_analysis.py --target activity --model lightgbm
    python source/scripts/factorial_analysis.py --target distortion --universe common
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.evaluation.compare import load_run, scores  # noqa: E402
from activity_detection.paths import EXPERIMENTS_DIR  # noqa: E402

# (detector, preprocessing, features) -> experiment folder
CELLS: Dict[Tuple[int, int, int], str] = {
    (0, 0, 0): "group_a_classical",
    (1, 0, 0): "ablation_1_detector_only",
    (0, 1, 0): "ablation_2_preprocessing_only",
    (0, 0, 1): "ablation_3_features_only",
    (1, 1, 0): "ablation_5_detector_preprocessing",
    (1, 0, 1): "ablation_6_detector_features",
    (0, 1, 1): "ablation_4_full_pipeline_fixed_mog2",
    (1, 1, 1): "group_i_adaptive_mog2",
}
FACTORS = ("detector", "preprocessing", "features")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="activity", choices=["activity", "distortion"])
    p.add_argument("--model", default="lightgbm", help="classifier column to score (same in all 8 runs)")
    p.add_argument("--universe", default="all", choices=["all", "common"])
    p.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy", "balanced_accuracy"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output folder (default: experiments/factorial_<target>)")
    return p.parse_args(argv)


def build_table(runs, target, universe):
    """DataFrame: video, true, and one prediction column per cell ('d1p0f1'); NaN = no prediction."""
    column = "activity" if target == "activity" else "distortion"
    if universe == "common":
        table = None
        for cell, run in runs.items():
            part = run.predictions[["video", "true", run.model]].rename(columns={run.model: _name(cell)})
            table = part if table is None else table.merge(part.drop(columns="true"), on="video", how="inner")
        return table
    base = None
    for cell, run in runs.items():
        if run.attempted is None:
            raise FileNotFoundError(f"{run.directory} has no videos_attempted.csv; re-run it or use --universe common.")
        part = run.attempted[["video", column]].rename(columns={column: "true"})
        base = part if base is None else base.merge(part[["video"]], on="video", how="inner")
    table = base
    for cell, run in runs.items():
        table = table.merge(run.predictions[["video", run.model]].rename(columns={run.model: _name(cell)}),
                            on="video", how="left")
    return table


def _name(cell):
    return "d{}p{}f{}".format(*cell)


def cell_scores(table, true_idx, classes, metric, idx):
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    out = {}
    for cell in CELLS:
        col = table[_name(cell)].to_numpy()
        pred = np.array([index[str(v)] if pd.notna(v) else k for v in col])
        out[cell] = scores(true_idx[idx], pred[idx], k)[metric]
    return out


def effects(c: Dict[Tuple[int, int, int], float]) -> Dict[str, float]:
    """Simple, main and interaction effects from the 8 cell scores."""
    out = {}
    for i, name in enumerate(FACTORS):
        others = [j for j in range(3) if j != i]
        diffs = []
        for levels in itertools.product((0, 1), repeat=2):
            lo, hi = [0, 0, 0], [0, 0, 0]
            for j, lv in zip(others, levels):
                lo[j] = hi[j] = lv
            lo[i], hi[i] = 0, 1
            d = c[tuple(hi)] - c[tuple(lo)]
            diffs.append(d)
            tag = ",".join(f"{FACTORS[j][0].upper()}={lv}" for j, lv in zip(others, levels))
            out[f"simple: {name} (1-0) at {tag}"] = d
        out[f"main: {name}"] = float(np.mean(diffs))
    for i, j in itertools.combinations(range(3), 2):
        k = 3 - i - j
        ix = []
        for lv in (0, 1):
            def eff(fi, lvj):
                lo, hi = [0] * 3, [0] * 3
                lo[k] = hi[k] = lv
                lo[j] = hi[j] = lvj
                hi[fi] = 1
                return c[tuple(hi)] - c[tuple(lo)]
            ix.append(eff(i, 1) - eff(i, 0))
        out[f"interaction: {FACTORS[i]} x {FACTORS[j]}"] = float(np.mean(ix))
    return out


def analyse(runs, target, universe, metric, n_boot, seed):
    table = build_table(runs, target, universe)
    if table is None or len(table) == 0:
        raise ValueError("The eight experiments have no videos in common.")
    classes = sorted(set(table["true"].astype(str)) |
                     {str(v) for cell in CELLS for v in table[_name(cell)].dropna()})
    index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([index[str(v)] for v in table["true"]])
    n = len(table)
    point = effects(cell_scores(table, true_idx, classes, metric, np.arange(n)))
    cells_point = cell_scores(table, true_idx, classes, metric, np.arange(n))
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in point}
    cell_draws = {c: np.empty(n_boot) for c in CELLS}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cs = cell_scores(table, true_idx, classes, metric, idx)
        for c, v in cs.items():
            cell_draws[c][b] = v
        for k, v in effects(cs).items():
            draws[k][b] = v
    rows = []
    for c in CELLS:
        lo, hi = np.percentile(cell_draws[c], [2.5, 97.5])
        rows.append({"kind": "cell", "name": f"{_name(c)}  {CELLS[c]}", "estimate": cells_point[c],
                     "ci_low": lo, "ci_high": hi, "excludes_0": ""})
    for k, est in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        rows.append({"kind": k.split(":")[0], "name": k.split(": ", 1)[1], "estimate": est,
                     "ci_low": lo, "ci_high": hi, "excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows), n, len(classes)


def plot_effects(df, path, title):
    """Forest plot: main effects and simple detector effects with 95% intervals."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the plot.")
        return
    sel = df[(df.kind == "main") | (df.kind == "interaction") |
             ((df.kind == "simple") & df.name.str.startswith("detector"))].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(sel) + 1.2))
    y = np.arange(len(sel))
    est, lo, hi = sel.estimate.to_numpy(), sel.ci_low.to_numpy(), sel.ci_high.to_numpy()
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}: {n}" for k, n in zip(sel.kind, sel.name)], fontsize=7)
    ax.set_xlabel("difference in score (95% bootstrap CI)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Plot saved: {path}")


def main(argv=None):
    a = parse_args(argv)
    runs = {cell: load_run(f"{name}:{a.model}", a.target, a.model, a.experiments_dir)
            for cell, name in CELLS.items()}
    df, n, k = analyse(runs, a.target, a.universe, a.metric, a.n_boot, a.seed)
    out = a.out or (a.experiments_dir / f"factorial_{a.target}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"factorial_{a.universe}_{a.model}.csv", index=False)
    plot_effects(df, out / f"factorial_{a.universe}_{a.model}.png",
                 f"{a.metric}, {a.target}, {a.model}, {a.universe} videos (n={n})")
    with open(out / f"factorial_{a.universe}_{a.model}.json", "w") as f:
        json.dump({"target": a.target, "model": a.model, "universe": a.universe, "metric": a.metric,
                   "n_videos": n, "n_classes": k, "n_boot": a.n_boot,
                   "runs": {_name(c): str(r.directory) for c, r in runs.items()}}, f, indent=2)
    pd.set_option("display.width", 160, "display.max_colwidth", 70)
    print(f"{a.metric}, target={a.target}, model={a.model}, universe={a.universe}, {n} videos\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"))
    print(f"\nSaved to {out}")
    return df


if __name__ == "__main__":
    main()
