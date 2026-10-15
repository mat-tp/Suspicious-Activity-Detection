import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import audit_duplicates as ad  # noqa: E402


def _video(path, seed, n=40, gain=1.0, noise=0.0):
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    x, y = rng.integers(0, 60), rng.integers(0, 60)
    vx, vy = rng.integers(1, 4), rng.integers(1, 4)
    speed = rng.uniform(0.5, 2.0, n)              # distinct motion profile per recording
    pos = 0.0
    bg = rng.integers(60, 120, (120, 160, 3)).astype(np.float32)
    for i in range(n):
        f = bg.copy()
        pos += speed[i]
        f[int(y + pos * vy) % 80:int(y + pos * vy) % 80 + 30, int(x + pos * vx) % 100:int(x + pos * vx) % 100 + 30] = 240
        f = np.clip(f * gain + (rng.normal(0, noise, f.shape) if noise else 0), 0, 255).astype(np.uint8)
        w.write(f)
    w.release()


def test_reencoded_copy_is_grouped_and_other_recording_is_not(tmp_path):
    _video(tmp_path / "a.avi", seed=1)
    _video(tmp_path / "a_lowq.avi", seed=1, gain=0.97, noise=2.0)   # same footage, degraded
    _video(tmp_path / "b.avi", seed=2)
    prints = {n: ad.fingerprint(tmp_path / n) for n in ["a.avi", "a_lowq.avi", "b.avi"]}
    g = ad.cluster(prints)
    assert g["a.avi"] == g["a_lowq.avi"] != g["b.avi"]


def test_frame_tolerance_and_size_mismatch_controls(tmp_path):
    base = {"frames": 40, "size": (160, 120), "curve": np.linspace(0, 1, 64),
            "thumb": np.arange(768, dtype=np.float32)}
    longer = {**base, "frames": 42}
    resized = {**base, "size": (320, 240)}
    assert not ad.same_recording(base, longer, 0.98, 0.9)
    assert ad.same_recording(base, longer, 0.98, 0.9, frame_tolerance=2)
    assert not ad.same_recording(base, resized, 0.98, 0.9)
    assert ad.same_recording(base, resized, 0.98, 0.9, require_same_size=False)
    groups = ad.cluster({"a": base, "b": longer}, frame_tolerance=2)
    assert groups["a"] == groups["b"]


def test_outputs_record_settings_checksum_and_unfingerprinted_videos(tmp_path):
    groups = {"a.avi": 0, "a2.avi": 0, "b.avi": 1}
    out = tmp_path / "groups.csv"
    meta = ad.write_outputs(out, groups, ["broken.avi"], {"motion_corr": 0.98}, 5, 42)
    assert meta["n_unfingerprinted"] == 1 and meta["n_groups"] == 3
    assert meta["groups_csv_sha256"] == ad.file_sha256(out)
    assert "broken.avi" in out.read_text()                 # every video keeps a row
    assert (tmp_path / "groups_meta.json").exists()
    assert "merged_group" in (tmp_path / "groups_review.csv").read_text()
