"""Find videos that are copies of the same recording, and write a group table.

The AD-SVD page says distorted videos were exported at three H.264 qualities, so
several files can contain the same footage. If copies fall on both sides of a
cross-validation split, scores are inflated. This script fingerprints each video
(frame count, resolution, a thumbnail of the middle frame and the motion-energy
curve over time) and merges videos whose fingerprints match into one group.

    python source/scripts/audit_duplicates.py --out data/groups.csv [--n 200]

Then set ``training.group_map_csv: data/groups.csv`` (see configs/grouped_cv_example.yaml).
Check the printed group-size histogram: if almost every group has size 1 the fingerprint
is too strict for your data; if one group swallows everything it is too loose (adjust
--motion-corr / --thumb-corr, or use --frame-tolerance / --allow-size-mismatch when copies
differ in length or resolution). Three files are written: the groups table, a review
listing (all merged groups plus sampled unmatched videos) and a metadata JSON with the
thresholds, the number of videos that could not be fingerprinted and the table's checksum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CURVE_LEN = 64


def fingerprint(path: Path) -> Optional[Dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    prev, diffs, thumb, i = None, [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24)).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        if i == max(n // 2, 0):
            thumb = small.flatten()
        prev, i = small, i + 1
    cap.release()
    if not diffs or thumb is None:
        return None
    curve = np.interp(np.linspace(0, len(diffs) - 1, CURVE_LEN), np.arange(len(diffs)), diffs)
    return {"frames": i, "size": size, "curve": curve, "thumb": thumb}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def same_recording(a: Dict, b: Dict, motion_corr: float, thumb_corr: float,
                   frame_tolerance: int = 0, require_same_size: bool = True) -> bool:
    """True when two fingerprints look like the same footage.

    frame_tolerance   allowed difference in frame count (re-encoding or frame-rate
                      conversion can change it); 0 demands equal counts.
    require_same_size set False to also match videos that were resized; thumbnails are
                      always compared at 32x24, so they stay comparable.
    """
    if abs(a["frames"] - b["frames"]) > frame_tolerance:
        return False
    if require_same_size and a["size"] != b["size"]:
        return False
    return (_corr(a["curve"], b["curve"]) >= motion_corr
            and _corr(a["thumb"], b["thumb"]) >= thumb_corr)


def cluster(prints: Dict[str, Dict], motion_corr: float = 0.98, thumb_corr: float = 0.9,
            frame_tolerance: int = 0, require_same_size: bool = True) -> Dict[str, int]:
    """Union-find over all matching pairs; returns {video name: group id}."""
    names = list(prints)
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # With exact frame count and size, only videos in the same bucket can match, which
    # keeps the pair count small. Otherwise every pair is compared.
    exact = frame_tolerance == 0 and require_same_size
    buckets: Dict[Tuple, List[int]] = {}
    for i, name in enumerate(names):
        key = (prints[name]["frames"], prints[name]["size"]) if exact else ("all",)
        buckets.setdefault(key, []).append(i)
    for indices in buckets.values():
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                if same_recording(prints[names[indices[x]]], prints[names[indices[y]]],
                                  motion_corr, thumb_corr, frame_tolerance, require_same_size):
                    parent[find(indices[x])] = find(indices[y])
    roots: Dict[int, int] = {}
    return {name: roots.setdefault(find(i), len(roots)) for i, name in enumerate(names)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(out: Path, groups: Dict[str, int], unfingerprinted: List[str], settings: Dict,
                  review_singletons: int, seed: int) -> Dict:
    """Write groups.csv, groups_review.csv and groups_meta.json (settings, counts, checksum).

    Videos that could not be fingerprinted get a group of their own, so every video in
    the dataset has a row (derive_groups needs that) and the count is reported.
    """
    all_groups = dict(groups)
    next_id = (max(groups.values()) + 1) if groups else 0
    for offset, name in enumerate(unfingerprinted):
        all_groups[name] = next_id + offset

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_name", "group"])
        writer.writerows(all_groups.items())

    # Review listing: every multi-video group, plus a random sample of singletons, so a
    # person can check for false merges and for missed duplicates.
    members: Dict[int, List[str]] = {}
    for name, group in groups.items():
        members.setdefault(group, []).append(name)
    singletons = sorted(g for g, v in members.items() if len(v) == 1)
    rng = np.random.default_rng(seed)
    sampled = set(rng.permutation(singletons)[:review_singletons].tolist()) if singletons else set()
    review_path = out.with_name(out.stem + "_review.csv")
    with open(review_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "group", "video_name"])
        for group, names in sorted(members.items()):
            if len(names) > 1:
                writer.writerows(("merged_group", group, n) for n in sorted(names))
            elif group in sampled:
                writer.writerow(("sampled_singleton", group, names[0]))

    sizes = Counter(Counter(all_groups.values()).values())
    meta = {
        "settings": settings,
        "n_videos": len(all_groups),
        "n_fingerprinted": len(groups),
        "n_unfingerprinted": len(unfingerprinted),
        "unfingerprinted_videos": unfingerprinted,
        "n_groups": len(set(all_groups.values())),
        "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "groups_csv": str(out),
        "groups_csv_sha256": file_sha256(out),
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    import pandas as pd
    from activity_detection.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="base")
    ap.add_argument("--out", default="data/groups.csv")
    ap.add_argument("--n", type=int, default=0, help="only the first N videos (0 = all)")
    ap.add_argument("--motion-corr", type=float, default=0.98)
    ap.add_argument("--thumb-corr", type=float, default=0.9)
    ap.add_argument("--frame-tolerance", type=int, default=0,
                    help="allowed difference in frame count between copies (default: equal)")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="also match videos whose resolution differs")
    ap.add_argument("--review-singletons", type=int, default=25,
                    help="how many unmatched videos to list for manual checking")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_excel(cfg.paths.dataset_excel)
    names = df["Name of Video Series"].tolist()
    names = names[: args.n] if args.n else names
    prints: Dict[str, Dict] = {}
    unfingerprinted: List[str] = []
    for k, name in enumerate(names, 1):
        fp = fingerprint(Path(cfg.paths.video_dir) / name)
        if fp:
            prints[name] = fp
        else:
            unfingerprinted.append(name)
        if k % 200 == 0:
            print(f"  fingerprinted {k}/{len(names)}")
    settings = {"motion_corr": args.motion_corr, "thumb_corr": args.thumb_corr,
                "frame_tolerance": args.frame_tolerance,
                "require_same_size": not args.allow_size_mismatch}
    groups = cluster(prints, args.motion_corr, args.thumb_corr, args.frame_tolerance,
                     not args.allow_size_mismatch)
    meta = write_outputs(Path(args.out), groups, unfingerprinted, settings,
                         args.review_singletons, args.seed)

    print(f"\n{meta['n_fingerprinted']} fingerprinted, {meta['n_unfingerprinted']} could not be "
          f"(each kept as its own group) -> {meta['n_groups']} groups; group-size histogram:")
    for size, count in meta["group_size_histogram"].items():
        print(f"  size {size}: {count} groups")
    print(f"Wrote {args.out} (sha256 {meta['groups_csv_sha256'][:12]}...), plus the review "
          f"listing and metadata next to it.")
    print("Check the review listing by eye before trusting grouped cross-validation.")


if __name__ == "__main__":
    main()
