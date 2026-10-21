"""Group labels for cross-validation that does not leak.

Several AD-SVD files can be re-encodings or re-distortions of the same recording. If one copy is in a
training fold and another in the test fold, the model is scored on something it has effectively seen, so
every copy of a recording must stay in one fold. The grouping is set in the YAML under ``training``:

    group_regex:   "^(.+?)_(?:Pri|Exp|Fo|ExFo)"   # first capture group of the file name
    group_map_csv: data/groups.csv                 # columns: video_name,group

With neither set, no groups are used and plain stratified k-fold runs.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def derive_groups(video_names: List[str], regex: Optional[str] = None,
                  map_csv: Optional[str] = None) -> Optional[np.ndarray]:
    """Return one integer group id per video, or None when no grouping is configured."""
    if not regex and not map_csv:
        return None

    labels: List[str] = []
    if map_csv:
        with open(map_csv, newline="") as f:
            mapping: Dict[str, str] = {r["video_name"]: r["group"] for r in csv.DictReader(f)}
        missing = [n for n in video_names if n not in mapping]
        if missing:
            raise ValueError(f"{len(missing)} videos are missing from {map_csv}, e.g. {missing[:3]}")
        labels = [mapping[n] for n in video_names]
    else:
        pattern = re.compile(regex)
        for name in video_names:
            m = pattern.search(Path(name).stem)
            if not m or not m.groups():
                raise ValueError(f"group_regex {regex!r} has no capture group match in {name!r}")
            labels.append(m.group(1))

    _, ids = np.unique(labels, return_inverse=True)
    return ids.astype(int)


def group_summary(groups: np.ndarray) -> str:
    """One-line description of a grouping: number of groups and min / median / max group size."""
    sizes = np.bincount(groups)
    return (f"{len(sizes)} groups, sizes min/median/max = "
            f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
