"""Check that the AD-SVD dataset is where the configs expect it (paths come from base.yaml).

    python source/scripts/check_dataset.py

Exit status 1 when anything is missing, so build.sh / build.bat can skip the stages that
need the videos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_detection.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config("base")
    excel, videos, truth = Path(cfg.paths.dataset_excel), Path(cfg.paths.video_dir), Path(cfg.paths.gt_dir)
    problems = []
    for label, path in (("dataset Excel", excel), ("video folder", videos), ("ground-truth folder", truth)):
        print(f"  {label:20s} {path}  ->  {'found' if path.exists() else 'MISSING'}")
        if not path.exists():
            problems.append(label)
    if not problems:
        n_videos = len(list(videos.glob("*.avi")))
        print(f"  {n_videos} .avi files, {len(list(truth.glob('*')))} ground-truth files")
        if n_videos == 0:
            problems.append("no .avi files in the video folder")
    if problems:
        print("\nThe dataset is not in place: " + ", ".join(problems))
        print("See docs/howToRun.md, step 2, for where AD-SVD goes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
