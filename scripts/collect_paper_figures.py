"""Copies the ablation plots to docs/manuscript/images/ under the names paper.tex expects.

    python scripts/collect_paper_figures.py [--model lightgbm]

Missing plots are reported and skipped; paper.tex shows a placeholder for a missing figure.
"""
import argparse
import shutil
import sys

import bootstrap  # noqa: F401  (puts source/ on the import path)
from activity_detection.paths import EXPERIMENTS_DIR, PROJECT_ROOT


def mapping(model, universe="all"):
    e = EXPERIMENTS_DIR
    m = {}
    for t in ("activity", "distortion"):
        m[f"fig_threshold_{t}.png"] = e / f"mog2_threshold_{t}.png"
        m[f"fig_factorial_{t}.png"] = e / f"factorial_{t}" / f"factorial_{universe}_{model}.png"
        for group in ("preprocessing", "detector", "adaptive", "tracker", "features",
                      "classifier", "imbalance"):
            m[f"fig_ablation_{group}_{t}.png"] = e / "parameter_ablation" / f"ablation_{group}_{t}.png"
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="lightgbm")
    p.add_argument("--universe", default="all", choices=["all", "common"])
    p.add_argument("--strict", action="store_true", help="exit with status 1 if any figure is missing")
    a = p.parse_args(argv)
    dest = PROJECT_ROOT / "docs" / "manuscript" / "images"
    dest.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name, src in mapping(a.model, a.universe).items():
        if src.exists():
            shutil.copy(src, dest / name)
            print(f"copied  {name}")
        else:
            missing += 1
            print(f"MISSING {name}  (expected {src})")
    return 1 if (missing and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
