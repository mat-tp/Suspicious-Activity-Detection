"""Printing and saving evaluation results."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


def print_summary(model_name: str, cv: Dict) -> None:
    """Print the headline scores and the per-class table for one model."""
    fs, ci, pooled = cv["fold_scores"], cv["ci95"], cv["pooled"]
    print(f"  Folds ({cv['n_splits']} x {cv['n_repeats']}), mean +/- std:")
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        print(f"    {name:18s} {fs[name]['mean']:.4f} +/- {fs[name]['std']:.4f}")
    print("  Pooled out-of-fold, with 95% bootstrap interval:")
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        lo, hi = ci[name]
        print(f"    {name:18s} {pooled[name]:.4f}  [{lo:.4f}, {hi:.4f}]")
    if pooled.get("roc_auc_ovr_macro") is not None:
        print(f"    {'roc_auc_ovr_macro':18s} {pooled['roc_auc_ovr_macro']:.4f}")
    print(f"  Overfit gap (train - test accuracy): {cv['overfit_gap']:.4f}")
    print(f"  {'class':<22s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>9s}")
    for name, s in pooled["per_class"].items():
        print(f"  {name:<22s}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}{s['support']:>9d}")


def write_per_class_csv(path: Path, per_model: Dict[str, Dict]) -> None:
    """One row per (model, class)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "precision", "recall", "f1", "support"])
        for model, cv in per_model.items():
            for cls, s in cv["pooled"]["per_class"].items():
                writer.writerow([model, cls, f"{s['precision']:.4f}", f"{s['recall']:.4f}",
                                 f"{s['f1']:.4f}", s["support"]])


def write_summary_csv(path: Path, per_model: Dict[str, Dict], imbalance: str) -> None:
    """One row per model with the headline scores and their intervals."""
    header = ["model", "imbalance", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
              "balanced_accuracy", "bal_acc_ci_low", "bal_acc_ci_high",
              "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
              "weighted_f1", "mcc", "roc_auc_ovr_macro", "fold_macro_f1_mean", "fold_macro_f1_std"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model, cv in per_model.items():
            p, ci, fs = cv["pooled"], cv["ci95"], cv["fold_scores"]
            auc = p.get("roc_auc_ovr_macro")
            writer.writerow([
                model, imbalance,
                f"{p['accuracy']:.4f}", *(f"{v:.4f}" for v in ci["accuracy"]),
                f"{p['balanced_accuracy']:.4f}", *(f"{v:.4f}" for v in ci["balanced_accuracy"]),
                f"{p['macro_f1']:.4f}", *(f"{v:.4f}" for v in ci["macro_f1"]),
                f"{p['weighted_f1']:.4f}", f"{p['mcc']:.4f}",
                "" if auc is None else f"{auc:.4f}",
                f"{fs['macro_f1']['mean']:.4f}", f"{fs['macro_f1']['std']:.4f}"])


def write_attempted_videos(path: Path, groups: Dict, ok_names: Iterable[str],
                           failures: Dict[str, str]) -> None:
    """Every video the run tried, and whether it produced a feature vector.

    Needed to compare experiments fairly: a video that produced no features
    cannot be classified, and that has to count against the experiment.
    """
    ok = set(ok_names)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "activity", "distortion", "status"])
        for (activity, distortion), names in groups.items():
            for name in names:
                status = "ok" if name in ok else f"failed: {failures.get(name, 'unknown')}"
                writer.writerow([name, activity, distortion, status])


def write_predictions(path: Path, video_names: List[str], true_labels: List[str],
                      predictions: Dict[str, List[str]]) -> None:
    """Out-of-fold prediction for every video, one column per model."""
    models = list(predictions)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "true", *models])
        for i, (name, true) in enumerate(zip(video_names, true_labels)):
            writer.writerow([name, true, *(predictions[m][i] for m in models)])
