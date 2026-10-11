"""Compare experiments, with differences and 95% intervals instead of t-tests.

Needs runs made with the current code (they save predictions_<target>.csv and
videos_attempted.csv). Names are experiment folders in experiments/; the latest
run is used. Append :model to pick a classifier, e.g. group_a_classical:svm_rbf.

Examples (from the project root):

    # does adaptive MOG2 alone help? (differences are candidate minus reference)
    python source/scripts/compare_experiments.py group_a_classical ablation_1_detector_only

    # several candidates against one reference, Holm-corrected
    python source/scripts/compare_experiments.py group_a_classical \\
        ablation_1_detector_only detector_vibe detector_pbas --model lightgbm

    # two classifiers on the same experiment
    python source/scripts/compare_experiments.py group_i_adaptive_mog2:svm_rbf \\
        group_i_adaptive_mog2:lightgbm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.evaluation.compare import (  # noqa: E402
    UNIVERSES,
    compare_many,
    format_per_class,
    format_result,
    load_run,
    per_class_table,
    results_table,
)
from activity_detection.paths import EXPERIMENTS_DIR  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", help="experiment (or experiment:model) to compare against")
    p.add_argument("candidates", nargs="+", help="experiments (or experiment:model) to compare")
    p.add_argument("--target", default="activity", choices=["activity", "distortion"])
    p.add_argument("--model", default="lightgbm",
                   help="classifier to use when a name has no :model (fix it in advance, "
                        "do not pick the best one per experiment)")
    p.add_argument("--universe", default="both", choices=[*UNIVERSES, "both"],
                   help="common = videos both predicted; all = every attempted video, "
                        "missing predictions count as misses")
    p.add_argument("--per-class", action="store_true", help="also print the per-class F1 differences")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    p.add_argument("--out", type=Path, help="output folder (default: experiments/comparisons/<reference>)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        reference = load_run(args.reference, args.target, args.model, args.experiments_dir)
        candidates = [load_run(c, args.target, args.model, args.experiments_dir) for c in args.candidates]
    except (FileNotFoundError, KeyError) as err:
        print(f"ERROR: {err}")
        return 1

    universes = list(UNIVERSES) if args.universe == "both" else [args.universe]
    out_dir = args.out or args.experiments_dir / "comparisons" / reference.label.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for universe in universes:
        try:
            results = compare_many(reference, candidates, args.target, universe, args.n_boot, args.seed)
        except FileNotFoundError as err:
            print(f"\nSkipping '{universe}': {err}")
            continue
        print(f"\n{'=' * 78}\n  Universe: {universe}\n{'=' * 78}")
        for r in results:
            print("\n" + format_result(r))
            if args.per_class:
                print(format_per_class(r))
        all_results += results

    if all_results:
        results_table(all_results).to_csv(out_dir / "comparison.csv", index=False)
        per_class_table(all_results).to_csv(out_dir / "comparison_per_class.csv", index=False)
        (out_dir / "comparison.json").write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved to {out_dir}")
        print("Intervals show sampling uncertainty over videos only; see evaluation/compare.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
