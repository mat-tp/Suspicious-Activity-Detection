"""Small helpers for experiment bookkeeping (run ids, git commit, reference trajectory length)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_next_run_id(experiment_dir: Path) -> str:
    """Next unused run id (``run_001``, ``run_002``, ...) among the ``run_*`` folders of *experiment_dir*."""
    existing_numbers = []
    if experiment_dir.exists():
        for d in experiment_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    existing_numbers.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    next_num = max(existing_numbers, default=0) + 1
    return f"run_{next_num:03d}"


def get_git_commit() -> str:
    """Short hash of the current git commit, or ``"unknown"`` outside a repository."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_min_trajectory_len(n_frames: int, base_min: int = 30) -> int:
    """
    Reference trajectory length, scaled to video duration.

    NOT a hard gate. Used only as the denominator in
    AdaptiveMOG2Detector.feedback() and for reporting.  The pipeline
    processes every track regardless of length; multi-family blending
    compensates when no tracks exist (Mauthner et al. 2009).
    """
    return min(base_min, max(10, int(n_frames * 0.6)))
