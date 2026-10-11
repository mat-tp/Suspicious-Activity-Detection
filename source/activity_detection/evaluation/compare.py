"""Compare experiments on the same videos, with differences and intervals.

Each experiment saves one out-of-fold prediction per video (predictions_<target>.csv)
and which videos it managed to process (videos_attempted.csv). Here two runs are
lined up video by video, and the difference in each metric is given with a 95%
paired-bootstrap interval: videos are resampled with replacement, the same videos
for both experiments, and the metric difference is recomputed each time.

Two sets of videos can be compared:

  common   videos that both experiments produced a prediction for
  all      every video both experiments attempted. A video with no prediction
           (no trajectories, so no features) counts as a miss. This is the fair
           end-to-end comparison when one detector loses many more videos.

What the intervals mean: they show how much a difference depends on which videos
were sampled. They do not include the randomness of retraining the models, and
folds share training data, so treat a small gap with caution even when its
interval excludes zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..paths import EXPERIMENTS_DIR

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
UNIVERSES = ("common", "all")


# ---- loading runs ---------------------------------------------------------

@dataclass
class Run:
    label: str                      # shown in the tables, e.g. "group_a_classical:lightgbm"
    directory: Path
    model: str
    predictions: pd.DataFrame       # columns: video, true, <model>...
    attempted: Optional[pd.DataFrame]   # columns: video, activity, distortion, status


def find_run_dir(name_or_path: str, target: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Path:
    """An experiment name gives its latest run that has predictions; a path is used as is."""
    path = Path(name_or_path)
    if path.is_dir() and (path / f"predictions_{target}.csv").exists():
        return path
    experiment = experiments_dir / name_or_path
    runs = sorted(d for d in experiment.glob("run_*") if (d / f"predictions_{target}.csv").exists())
    if not runs:
        raise FileNotFoundError(
            f"No run of '{name_or_path}' has predictions_{target}.csv in {experiments_dir}. "
            "Runs made before this feature need to be run again.")
    return runs[-1]


def load_run(spec: str, target: str, default_model: str,
             experiments_dir: Path = EXPERIMENTS_DIR) -> Run:
    """spec is 'experiment' or 'experiment:model'."""
    name, _, model = spec.partition(":")
    model = model or default_model
    directory = find_run_dir(name, target, experiments_dir)
    predictions = pd.read_csv(directory / f"predictions_{target}.csv")
    if model not in predictions.columns:
        raise KeyError(f"{directory.name} of '{name}' has no model '{model}'. "
                       f"Available: {[c for c in predictions.columns if c not in ('video', 'true')]}")
    attempted_file = directory / "videos_attempted.csv"
    attempted = pd.read_csv(attempted_file) if attempted_file.exists() else None
    return Run(label=f"{name}:{model}", directory=directory, model=model,
               predictions=predictions, attempted=attempted)


# ---- lining up two runs ---------------------------------------------------

@dataclass
class Pair:
    classes: List[str]
    true: np.ndarray        # class index per video
    pred_a: np.ndarray      # class index, or len(classes) for "no prediction"
    pred_b: np.ndarray
    counts: Dict[str, int]


def align(a: Run, b: Run, target: str, universe: str) -> Pair:
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}")
    pa = a.predictions[["video", "true", a.model]].rename(columns={a.model: "pred_a"})
    pb = b.predictions[["video", b.model]].rename(columns={b.model: "pred_b"})

    if universe == "common":
        merged = pa.merge(pb, on="video", how="inner")
        true = merged["true"]
    else:
        if a.attempted is None or b.attempted is None:
            raise FileNotFoundError("Comparing on all videos needs videos_attempted.csv in both runs. "
                                    "Run the experiments again, or use --universe common.")
        column = "activity" if target == "activity" else "distortion"
        universe_videos = a.attempted[["video", column]].merge(
            b.attempted[["video"]], on="video", how="inner").rename(columns={column: "true"})
        merged = universe_videos.merge(pa[["video", "pred_a"]], on="video", how="left") \
                                .merge(pb, on="video", how="left")
        true = merged["true"]

    if len(merged) == 0:
        raise ValueError("The two experiments have no videos in common.")

    seen = set(true.astype(str)) | set(merged["pred_a"].dropna().astype(str)) | set(merged["pred_b"].dropna().astype(str))
    classes = sorted(seen)
    index = {c: i for i, c in enumerate(classes)}
    reject = len(classes)
    to_index = lambda col: np.array([index[str(v)] if pd.notna(v) else reject for v in col])
    return Pair(
        classes=classes, true=to_index(true), pred_a=to_index(merged["pred_a"]),
        pred_b=to_index(merged["pred_b"]),
        counts={"videos": len(merged),
                "evaluated_a": int((merged["pred_a"].notna()).sum()),
                "evaluated_b": int((merged["pred_b"].notna()).sum())})


# ---- scores ---------------------------------------------------------------

def scores(true: np.ndarray, pred: np.ndarray, k: int) -> Dict[str, object]:
    """Metrics where prediction index k means 'no prediction' (always a miss)."""
    cm = np.bincount(true * (k + 1) + pred, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    tp = np.diag(cm)[:k].astype(float)
    support = cm[:k].sum(axis=1).astype(float)
    predicted = cm[:, :k].sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / max(len(true), 1)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1.mean()),
        "f1": f1, "recall": recall,
    }


def paired_bootstrap(pair: Pair, n_boot: int = 2000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Differences (b minus a) over bootstrap resamples of the videos."""
    k = len(pair.classes)
    n = len(pair.true)
    rng = np.random.default_rng(seed)
    out = {m: np.empty(n_boot) for m in METRICS}
    out["f1"] = np.empty((n_boot, k))
    out["recall"] = np.empty((n_boot, k))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sa = scores(pair.true[idx], pair.pred_a[idx], k)
        sb = scores(pair.true[idx], pair.pred_b[idx], k)
        for m in METRICS:
            out[m][i] = sb[m] - sa[m]
        out["f1"][i] = sb["f1"] - sa["f1"]
        out["recall"][i] = sb["recall"] - sa["recall"]
    return out


def interval(draws: np.ndarray, alpha: float = 0.05) -> List[float]:
    return [float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))]


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "B higher"
    if high < 0:
        return "A higher"
    return "interval includes 0"


def mcnemar_exact(pair: Pair) -> Dict[str, float]:
    """Exact McNemar test on videos where exactly one experiment was right."""
    correct_a = pair.pred_a == pair.true
    correct_b = pair.pred_b == pair.true
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": float(p)}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, for comparing several candidates to one reference."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ---- one comparison -------------------------------------------------------

def compare_runs(a: Run, b: Run, target: str, universe: str,
                 n_boot: int = 2000, seed: int = 0) -> Dict:
    """Compare run B against run A (differences are B minus A)."""
    pair = align(a, b, target, universe)
    k = len(pair.classes)
    sa, sb = scores(pair.true, pair.pred_a, k), scores(pair.true, pair.pred_b, k)
    draws = paired_bootstrap(pair, n_boot, seed)

    metrics = {}
    for m in METRICS:
        low, high = interval(draws[m])
        metrics[m] = {"a": sa[m], "b": sb[m], "diff": sb[m] - sa[m],
                      "ci_low": low, "ci_high": high, "verdict": verdict(low, high)}
    per_class = {}
    for i, name in enumerate(pair.classes):
        f_low, f_high = interval(draws["f1"][:, i])
        r_low, r_high = interval(draws["recall"][:, i])
        per_class[name] = {
            "support": int((pair.true == i).sum()),
            "f1_a": float(sa["f1"][i]), "f1_b": float(sb["f1"][i]),
            "f1_diff": float(sb["f1"][i] - sa["f1"][i]), "f1_ci_low": f_low, "f1_ci_high": f_high,
            "recall_a": float(sa["recall"][i]), "recall_b": float(sb["recall"][i]),
            "recall_diff": float(sb["recall"][i] - sa["recall"][i]),
            "recall_ci_low": r_low, "recall_ci_high": r_high}
    return {"a": a.label, "b": b.label, "target": target, "universe": universe,
            "counts": pair.counts, "n_boot": n_boot, "metrics": metrics,
            "mcnemar": mcnemar_exact(pair), "per_class": per_class}


def compare_many(reference: Run, candidates: Sequence[Run], target: str, universe: str,
                 n_boot: int = 2000, seed: int = 0) -> List[Dict]:
    """Each candidate against the reference, with Holm-adjusted McNemar p-values."""
    results = [compare_runs(reference, c, target, universe, n_boot, seed) for c in candidates]
    adjusted = holm_adjust([r["mcnemar"]["p_value"] for r in results])
    for r, p in zip(results, adjusted):
        r["mcnemar"]["p_holm"] = p
    return results


# ---- output ---------------------------------------------------------------

def format_result(r: Dict) -> str:
    c = r["counts"]
    lines = [f"{r['a']}  ->  {r['b']}    [{r['target']}, {r['universe']} videos: {c['videos']}]",
             f"  predictions made:  A {c['evaluated_a']}   B {c['evaluated_b']}"]
    lines.append(f"  {'metric':<19}{'A':>8}{'B':>8}{'B - A':>9}   95% interval          ")
    for m in METRICS:
        v = r["metrics"][m]
        lines.append(f"  {m:<19}{v['a']:>8.4f}{v['b']:>8.4f}{v['diff']:>+9.4f}   "
                     f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  {v['verdict']}")
    t = r["mcnemar"]
    holm = f", Holm-adjusted {t['p_holm']:.4g}" if "p_holm" in t else ""
    lines.append(f"  McNemar (accuracy): only A right {t['a_only_correct']}, "
                 f"only B right {t['b_only_correct']}, p = {t['p_value']:.4g}{holm}")
    return "\n".join(lines)


def format_per_class(r: Dict) -> str:
    lines = [f"  {'class':<22}{'support':>8}{'F1 A':>8}{'F1 B':>8}{'diff':>9}   95% interval"]
    for name, v in r["per_class"].items():
        lines.append(f"  {name:<22}{v['support']:>8d}{v['f1_a']:>8.3f}{v['f1_b']:>8.3f}"
                     f"{v['f1_diff']:>+9.3f}   [{v['f1_ci_low']:+.3f}, {v['f1_ci_high']:+.3f}]")
    return "\n".join(lines)


def results_table(results: Sequence[Dict]) -> pd.DataFrame:
    """One row per comparison, for saving as CSV."""
    rows = []
    for r in results:
        row = {"a": r["a"], "b": r["b"], "target": r["target"], "universe": r["universe"],
               **{f"n_{k}": v for k, v in r["counts"].items()}}
        for m in METRICS:
            v = r["metrics"][m]
            row.update({f"{m}_a": v["a"], f"{m}_b": v["b"], f"{m}_diff": v["diff"],
                        f"{m}_ci_low": v["ci_low"], f"{m}_ci_high": v["ci_high"]})
        row.update({f"mcnemar_{k}": v for k, v in r["mcnemar"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(results: Sequence[Dict]) -> pd.DataFrame:
    rows = [{"a": r["a"], "b": r["b"], "universe": r["universe"], "class": name, **v}
            for r in results for name, v in r["per_class"].items()]
    return pd.DataFrame(rows)
