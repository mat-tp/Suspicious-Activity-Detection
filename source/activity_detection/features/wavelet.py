from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..interfaces import FeatureDict, FeatureExtractor, Track
from ..optional import PYWAVELETS_AVAILABLE, pywt
from ..registry import register


@register("feature", "wavelet")
class WaveletFeatures(FeatureExtractor):
    def __init__(self, wavelet="db4", level=3, min_trajectory_len=30):
        self.wavelet = wavelet
        self.level = level
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["dwt_energy_approx", "dwt_energy_detail1", "dwt_energy_detail2",
                "dwt_energy_detail3", "dwt_energy_total", "dwt_entropy"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
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
            speeds = np.pad(speeds, (0, pad_len), mode="reflect")
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
