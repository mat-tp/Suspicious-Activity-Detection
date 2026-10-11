"""Copy the ablation plots into docs/images/ under the names paper.tex expects.

Run after the experiments:   python source/scripts/collect_paper_figures.py [--model lightgbm]
Missing plots are reported and skipped; paper.tex shows a red placeholder for any
figure that is not there yet, so it always compiles.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity_detection.paths import EXPERIMENTS_DIR, PROJECT_ROOT  # noqa: E402


def mapping(model, universe="all"):
    e = EXPERIMENTS_DIR
    m = {}
    for t in ("activity", "distortion"):
        m[f"fig_threshold_{t}.png"] = e / f"mog2_threshold_{t}.png"
        m[f"fig_factorial_{t}.png"] = e / f"factorial_{t}" / f"factorial_{universe}_{model}.png"
        m[f"fig_sensitivity_{t}.png"] = e / f"sensitivity_ablation_1_detector_only_{t}" / "sensitivity.png"
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="lightgbm")
    p.add_argument("--universe", default="all", choices=["all", "common"])
    a = p.parse_args(argv)
    dest = PROJECT_ROOT / "docs" / "images"
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in mapping(a.model, a.universe).items():
        if src.exists():
            shutil.copy(src, dest / name)
            print(f"copied  {name}")
        else:
            print(f"MISSING {name}  (expected {src})")


if __name__ == "__main__":
    main()
