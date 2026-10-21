"""Writes a table that says what every feature column is (family and description).

Without arguments it lists every feature the code can produce. With ``--csv`` it lists the
columns of a ``video_features.csv`` and reports the ones the catalogue does not know.

    python scripts/list_features.py
    python scripts/list_features.py --csv experiments/<run>/video_features.csv
"""
import argparse
import csv
import sys

import bootstrap  # noqa: F401  (puts source/ on the import path)
from activity_detection.features.catalogue import describe
from activity_detection.features.interaction import InteractionFeatures
from activity_detection.features.track_features import (
    CircularMotionFeatures, KinematicsFeatures, ShapeFeatures, TrajectoryFeatures)
from activity_detection.features.video_features import (
    extract_appearance_features, extract_diagnostic_features, extract_global_flow_features,
    extract_temporal_context)
from activity_detection.features.wavelet import WaveletFeatures

# These are the feature extractors whose descriptors are computed per track.
TRACK_MODULES = (TrajectoryFeatures, CircularMotionFeatures, KinematicsFeatures, ShapeFeatures,
                 WaveletFeatures, InteractionFeatures)
# These are the columns of video_features.csv that are labels, not features.
NOT_FEATURES = {"video_name", "activity", "distortion", "label"}


def all_feature_names():
    """Every column the full pipeline can produce."""
    names = [f"{agg}_{descriptor}" for agg in ("mean", "max", "std")
             for cls in TRACK_MODULES for descriptor in cls().feature_names()]
    names += ["n_valid_tracks", "indicator_tracks_present", "meta_n_traj_tracks",
              "meta_num_frames", "meta_duration_sec", "meta_fps"]
    names += list(extract_diagnostic_features([])) + list(extract_global_flow_features([]))
    names += list(extract_appearance_features([]))
    names += list(extract_temporal_context([{"trajectory": [(0, 0)] * 20}] * 2, 40))
    return names


def csv_columns(path):
    with open(path, newline="") as handle:
        header = next(csv.reader(handle))
    return [column for column in header if column not in NOT_FEATURES]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", help="a video_features.csv to describe")
    parser.add_argument("--out", default="feature_catalogue.csv", help="where to write the table")
    args = parser.parse_args()

    names = csv_columns(args.csv) if args.csv else all_feature_names()
    unknown = []
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "family", "description"])
        for name in names:
            info = describe(name)
            if info is None:
                unknown.append(name)
                writer.writerow([name, "unknown", ""])
            else:
                writer.writerow([name, info.family, info.description])
    print(f"{len(names)} features written to {args.out}")
    if unknown:
        print(f"{len(unknown)} not in the catalogue: {', '.join(unknown[:10])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
