"""The paper's feature appendix must list every descriptor the code can produce."""
import re
from pathlib import Path

import activity_detection.features.interaction  # noqa: F401  (registers extractors)
import activity_detection.features.track_features  # noqa: F401
import activity_detection.features.wavelet  # noqa: F401
from activity_detection.features.interaction import InteractionFeatures
from activity_detection.features.track_features import (
    CircularMotionFeatures, KinematicsFeatures, ShapeFeatures, TrajectoryFeatures)
from activity_detection.features.video_features import (
    extract_appearance_features, extract_diagnostic_features, extract_global_flow_features,
    extract_temporal_context)
from activity_detection.features.wavelet import WaveletFeatures

APPENDIX = (Path(__file__).resolve().parents[2] / "docs" / "manuscript" / "feature_appendix.tex").read_text()
LISTED = set(re.findall(r"\\f\{([^}]*)\}", APPENDIX))


def _listed(name: str) -> bool:
    if name in LISTED:
        return True
    for pat in LISTED:                       # wildcard entries: diag_laplacian_*, app_lbp_mean_0..9
        if pat.endswith("*") and name.startswith(pat[:-1]):
            return True
        m = re.fullmatch(r"(.*?)0\.\.(\d+)", pat)
        if m and re.fullmatch(re.escape(m.group(1)) + r"\d+", name):
            return True
    return False


def test_every_per_track_descriptor_is_documented():
    names = []
    for cls in (TrajectoryFeatures, CircularMotionFeatures, KinematicsFeatures, ShapeFeatures,
                WaveletFeatures, InteractionFeatures):
        names += cls().feature_names()
    assert len(names) == 45
    missing = [n for n in names if not _listed(n)]
    assert not missing, f"undocumented descriptors: {missing}"


def test_every_video_level_feature_is_documented():
    names = list(extract_diagnostic_features([])) + list(extract_global_flow_features([])) \
        + list(extract_appearance_features([]))
    assert (len(extract_diagnostic_features([])), len(extract_global_flow_features([])),
            len(extract_appearance_features([]))) == (16, 22, 34)
    names += ["n_valid_tracks", "meta_num_frames", "meta_duration_sec", "meta_fps",
              "indicator_tracks_present", "meta_n_traj_tracks"]
    tracks = [{"trajectory": [(0, 0)] * 20}, {"trajectory": [(1, 1)] * 20}]
    names += list(extract_temporal_context(tracks, 40))
    missing = [n for n in names if not _listed(n)]
    assert not missing, f"undocumented features: {missing}"


def test_dimension_arithmetic_matches_the_paper():
    assert 14 * 3 + 1 + 8 + 3 == 54
    assert 45 * 3 + 1 + 8 + 3 == 147
    assert 147 + 16 + 22 + 34 + 2 == 221
