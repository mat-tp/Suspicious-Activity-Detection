"""Share of the model importance that each feature family carries.

It reads the ``feature_importance_<target>.json`` of a run and sums the importance per family, so the
discussion can say what the classifier relies on.

    python scripts/feature_family_importance.py experiments/<run>/feature_importance_distortion.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

# Order matters: first matching prefix wins. Pooled per-track features are named
# "<mean|max|std>_<descriptor>", so they are matched on the descriptor.
IMAGE_PREFIXES = (("diag_", "image quality"), ("flow_", "global optical flow"),
                  ("app_", "appearance"), ("temporal_", "temporal context"),
                  ("meta_num_", "file metadata"), ("meta_duration", "file metadata"),
                  ("meta_fps", "file metadata"), ("meta_n_traj", "track count"),
                  ("n_valid_tracks", "track count"), ("indicator_", "track count"))
# These are the families that the per-track descriptors are grouped into.
DESCRIPTOR_FAMILIES = (
    ("circular", ("circular_", "direction_concentration", "angular_dispersion")),
    ("kinematic", ("accel", "jerk")),
    ("wavelet", ("dwt_",)),
    ("interaction", ("pairwise_distance", "relative_speed", "proximity_")),
    ("shape", ("curvature", "turning_rate", "convex_hull", "compactness", "path_efficiency")),
    ("trajectory", ("track_duration", "speed", "straightness", "direction", "bbox_area", "flow_magnitude",
                    "flow_angle")),
)


def family(feature: str) -> str:
    for prefix, name in IMAGE_PREFIXES:
        if feature.startswith(prefix):
            return name
    core = feature.split("_", 1)[1] if feature.split("_", 1)[0] in ("mean", "max", "std") else feature
    for name, keys in DESCRIPTOR_FAMILIES:
        if any(k in core for k in keys):
            return name
    return "other"


def by_family(importances: Dict[str, float]) -> Dict[str, float]:
    total = sum(importances.values()) or 1.0
    out: Dict[str, float] = defaultdict(float)
    for feat, value in importances.items():
        out[family(feat)] += value / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(path: str) -> None:
    data = json.loads(Path(path).read_text())
    for model, imp in data.items():
        print(f"\n{model}")
        for fam, share in by_family(imp).items():
            print(f"  {fam:<20s}{share:7.1%}")
        top = sorted(imp, key=imp.get, reverse=True)[:5]
        print("  top 5:", ", ".join(f"{t} ({imp[t]:.3f})" for t in top))


def latest_importance_file(experiment: str, target: str, experiments_dir: Path) -> Path:
    """feature_importance_<target>.json of the newest run of an experiment."""
    runs = sorted(d for d in (experiments_dir / experiment).glob("run_*")
                  if (d / f"feature_importance_{target}.json").exists())
    if not runs:
        raise SystemExit(f"No run of '{experiment}' in {experiments_dir} has "
                         f"feature_importance_{target}.json")
    return runs[-1] / f"feature_importance_{target}.json"


if __name__ == "__main__":
    import argparse

    from activity_detection.paths import EXPERIMENTS_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="a feature_importance_<target>.json file")
    parser.add_argument("--latest", metavar="EXPERIMENT",
                        help="use the newest run of this experiment instead of a path")
    parser.add_argument("--target", default="activity", choices=["activity", "distortion"])
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()
    if args.latest:
        main(str(latest_importance_file(args.latest, args.target, args.experiments_dir)))
    elif args.path:
        main(args.path)
    else:
        parser.error("give a path or --latest EXPERIMENT")
