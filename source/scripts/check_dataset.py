"""Check that the AD-SVD dataset is where the configs expect it (paths come from base.yaml).

    python source/scripts/check_dataset.py

The real test is whether the video names listed in datasetInfo.xlsx ("Name of Video Series")
exist in the video folder, because that is exactly what the experiments look up. File
extensions found in the folder are printed to help diagnose a wrong path or extension.
Exit status 1 when anything is missing, so build.sh / build.bat can skip the stages that
need the videos.
"""
import sys
from collections import Counter
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
    if problems:
        print("\nThe dataset is not in place: " + ", ".join(problems))
        print("See docs/howToRun.md, step 2, for where AD-SVD goes.")
        return 1

    extensions = Counter(p.suffix.lower() or "(none)" for p in videos.iterdir() if p.is_file())
    subfolders = [p.name for p in videos.iterdir() if p.is_dir()]
    print(f"  files in the video folder by extension: {dict(extensions) or 'none'}")
    if subfolders:
        print(f"  sub-folders in the video folder (videos must be directly inside it): {subfolders[:5]}")
    print(f"  {len(list(truth.glob('*')))} ground-truth files")

    import pandas as pd
    names = pd.read_excel(excel)["Name of Video Series"].astype(str).tolist()
    missing = [n for n in names if not (videos / n).exists()]
    print(f"  videos listed in the Excel file: {len(names)}; found on disk: {len(names) - len(missing)}")
    if missing:
        print(f"  first missing names: {missing[:5]}")
        print("  Check the folder path in source/configs/base.yaml and that the names match exactly "
              "(extension and upper/lower case included).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
