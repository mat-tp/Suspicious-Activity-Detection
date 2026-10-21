"""Wavelet energy descriptors of a track's speed signal."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..optional import PYWAVELETS_AVAILABLE, pywt
from ..registry import register


@register("feature", "wavelet")
class WaveletFeatures(FeatureExtractor):
    """Wavelet energy of the speed-over-time signal of a track.

    The speed is padded to a power-of-two length and decomposed with ``pywt.wavedec``. The output is the
    energy of the approximation and of detail levels 1-3, their total, and the entropy of the energy
    distribution. Needs PyWavelets and returns ``None`` without it.

    Args:
        wavelet: wavelet name, e.g. ``"db4"``.
        level: decomposition depth.
        min_trajectory_len: shorter tracks return ``None``.
    """

    def __init__(self, wavelet="db4", level=3, min_trajectory_len=30):
        self.wavelet = wavelet
        self.level = level
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["dwt_energy_approx", "dwt_energy_detail1", "dwt_energy_detail2",
                "dwt_energy_detail3", "dwt_energy_total", "dwt_entropy"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """Energy and entropy of the speed signal of one track; ``None`` if it is too short or PyWavelets is missing.

        One frame is assumed per trajectory step (see ``track_features`` for the caveat about missed frames).
        """
        if not PYWAVELETS_AVAILABLE:
            return None
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        fps    = float(context.get("fps", 10.0))
        deltas = np.diff(traj, axis=0)
        # Speeds in px/s, so wavelet energies are in (px/s)^2 and comparable across
        # videos with different frame rates. The transform is applied to the speed
        # signal of the track.
        speeds = np.linalg.norm(deltas, axis=1) * fps   # px/s
        n = len(speeds)
        pad_len = 2 ** int(np.ceil(np.log2(max(n, 8)))) - n
        if pad_len > 0:
            # "reflect" needs pad_len < n; very short signals are padded with their edge value.
            speeds = np.pad(speeds, (0, pad_len), mode="reflect" if pad_len < n else "edge")
        try:
            coeffs = pywt.wavedec(speeds, self.wavelet, level=self.level)
        except Exception:
            return None
        energies = [float(np.sum(np.array(c) ** 2)) for c in coeffs]
        total_energy = sum(energies)
        eps = 1e-10
        energy_dist = np.array(energies) / (total_energy + eps)
        entropy = float(-np.sum(energy_dist * np.log(energy_dist + eps)))
        result = {"dwt_energy_approx": energies[0] if energies else 0.0}
        for i in range(1, 4):
            result[f"dwt_energy_detail{i}"] = energies[i] if i < len(energies) else 0.0
        result["dwt_energy_total"] = total_energy
        result["dwt_entropy"] = entropy
        return result
