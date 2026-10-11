from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..features.aggregation import FeatureAggregator
from ..interfaces import FeatureDict, Track


def extract_temporal_context(tracks: List[Track], total_frames: int = 0) -> FeatureDict:
    """
    Compute video-level features that capture *when* and *where* tracks were
    active — information lost by per-track mean/max/std aggregation.

    These features are merged directly into the video feature vector alongside
    the aggregated track features so that models can distinguish activities
    characterised by concurrent multi-person motion (FG, PPP) from those that
    are typically single-person (LPP, PO).

    Args:
        tracks:       Active tracks returned by the tracker after video processing.
        total_frames: Number of frames processed (used to normalise coverage).

    Returns:
        FeatureDict with keys prefixed ``temporal_``.
    """
    if not tracks:
        return {}

    # Approximate start / end frame from trajectory length (we don't store
    # absolute frame indices, so we use cumulative order as a proxy)
    traj_lengths = [len(t["trajectory"]) for t in tracks]
    n_tracks = len(tracks)

    # Temporal spread: std of trajectory lengths approximates spread of when
    # tracks were active (long tracks span more of the video)
    temporal_spread = float(np.std(traj_lengths)) if n_tracks > 1 else 0.0

    # Spatial spread: std of all trajectory centroids across all tracks
    all_centroids = np.array(
        [pt for t in tracks for pt in t["trajectory"]], dtype=float
    )
    spatial_spread = float(np.std(all_centroids, axis=0).mean()) if len(all_centroids) > 1 else 0.0

    # Track overlap: naive estimate — count pairs where both are long enough
    # to plausibly co-exist (both ≥ 15 frames)
    long_tracks = [t for t in tracks if len(t["trajectory"]) >= 15]
    n_long = len(long_tracks)
    overlap_count = max(0, n_long * (n_long - 1) // 2)

    # Track density: tracks per frame
    tracks_per_frame = (sum(traj_lengths) / total_frames) if total_frames > 0 else 0.0

    return {
        "temporal_n_tracks": float(n_tracks),
        "temporal_n_long_tracks": float(n_long),
        "temporal_spread": temporal_spread,
        "temporal_spatial_spread": spatial_spread,
        "temporal_overlap_count": float(overlap_count),
        "temporal_tracks_per_frame": tracks_per_frame,
        "temporal_mean_track_len": float(np.mean(traj_lengths)) if traj_lengths else 0.0,
        "temporal_max_track_len": float(max(traj_lengths)) if traj_lengths else 0.0,
    }


def _pool_stat(arr: np.ndarray, stat: str) -> float:
    if arr is None or len(arr) == 0:
        return 0.0
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min":  float(np.min(arr)),  "max": float(np.max(arr))}.get(stat, 0.0)


def extract_diagnostic_features(frame_diagnostics: List[dict]) -> FeatureDict:
    """16 pooled diagnostic features — always available, no detection needed.
    Research: Gillani et al. (2025), Gomez-Nieto et al. (2022), Bai & Reibman (2010)."""
    _Z: FeatureDict = {k: 0.0 for k in [
        "diag_laplacian_mean","diag_laplacian_std","diag_laplacian_min","diag_laplacian_max",
        "diag_tenengrad_mean","diag_tenengrad_std","diag_tenengrad_min","diag_tenengrad_max",
        "diag_brenner_mean",  "diag_brenner_std",  "diag_brenner_min",  "diag_brenner_max",
        "diag_brightness_mean","diag_brightness_var","diag_dynamic_range","diag_low_contrast_ratio",
    ]}
    if not frame_diagnostics:
        return _Z
    laps    = np.array([d.get("laplacian_var", d.get("blur_score", 0.0)) for d in frame_diagnostics], dtype=float)
    tenens  = np.array([d.get("tenengrad",  0.0) for d in frame_diagnostics], dtype=float)
    brens   = np.array([d.get("brenner",    0.0) for d in frame_diagnostics], dtype=float)
    brights = np.array([d.get("brightness", 0.0) for d in frame_diagnostics], dtype=float)
    return {
        "diag_laplacian_mean":     _pool_stat(laps,    "mean"),
        "diag_laplacian_std":      _pool_stat(laps,    "std"),
        "diag_laplacian_min":      _pool_stat(laps,    "min"),
        "diag_laplacian_max":      _pool_stat(laps,    "max"),
        "diag_tenengrad_mean":     _pool_stat(tenens,  "mean"),
        "diag_tenengrad_std":      _pool_stat(tenens,  "std"),
        "diag_tenengrad_min":      _pool_stat(tenens,  "min"),
        "diag_tenengrad_max":      _pool_stat(tenens,  "max"),
        "diag_brenner_mean":       _pool_stat(brens,   "mean"),
        "diag_brenner_std":        _pool_stat(brens,   "std"),
        "diag_brenner_min":        _pool_stat(brens,   "min"),
        "diag_brenner_max":        _pool_stat(brens,   "max"),
        "diag_brightness_mean":    _pool_stat(brights, "mean"),
        "diag_brightness_var":     float(np.var(brights)) if len(brights) > 1 else 0.0,
        "diag_dynamic_range":      float(brights.max()-brights.min()) if len(brights) > 0 else 0.0,
        "diag_low_contrast_ratio": float(np.mean(laps < 50.0)) if len(laps) > 0 else 0.0,
    }


def extract_global_flow_features(gray_frames: List[np.ndarray],
                                  flow_step: int = 5) -> FeatureDict:
    """22 Farnebäck dense-flow features — always available, no detection needed.
    Research: Mauthner et al. (2009), Xiao et al. (2016), Saleh et al. (2022)."""
    _Z: FeatureDict = {
        "flow_mean_mag":0.0,"flow_std_mag":0.0,"flow_max_mag":0.0,
        "flow_prop_moving":0.0,"flow_dir_entropy":0.0,"flow_jerk":0.0,
        **{f"flow_hof_mean_{i}":0.0 for i in range(8)},
        **{f"flow_hof_std_{i}": 0.0 for i in range(8)},
    }
    if len(gray_frames) < 2:
        return _Z
    def _u8g(f):
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.shape[2] >= 3 else f[:,:,0]
        return f.astype(np.uint8)
    TH, TW = 120, 160
    mags, props, hofs = [], [], []
    prev_s = cv2.resize(_u8g(gray_frames[0]), (TW, TH))
    for idx in range(1, len(gray_frames), max(1, flow_step)):
        curr_s = cv2.resize(_u8g(gray_frames[idx]), (TW, TH))
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_s, curr_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        except cv2.error:
            prev_s = curr_s; continue
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(float(mag.mean()))
        props.append(float((mag > 1.0).mean()))
        if mag.sum() > 0:
            h, _ = np.histogram(ang[mag > 1.0], bins=8, range=(0, 2*np.pi))
            h = h.astype(float) / (h.sum()+1e-7)
        else:
            h = np.zeros(8)
        hofs.append(h); prev_s = curr_s
    if not mags:
        return _Z
    ma = np.array(mags); ha = np.array(hofs) if hofs else np.zeros((1,8))
    mh = ha.mean(0); sh = ha.std(0) if len(ha) > 1 else np.zeros(8)
    p  = mh / (mh.sum()+1e-7)
    ent = float(-np.sum(p[p > 0]*np.log(p[p > 0]+1e-12)))
    res: FeatureDict = {
        "flow_mean_mag":    float(ma.mean()), "flow_std_mag":      float(ma.std()),
        "flow_max_mag":     float(ma.max()),  "flow_prop_moving":  float(np.mean(props)),
        "flow_dir_entropy": ent,              "flow_jerk":         float(np.mean(np.abs(np.diff(ma)))) if len(ma) > 1 else 0.0,
    }
    for i in range(8):
        res[f"flow_hof_mean_{i}"] = float(mh[i])
        res[f"flow_hof_std_{i}"]  = float(sh[i])
    return res


def extract_appearance_features(gray_frames: List[np.ndarray]) -> FeatureDict:
    """34 LBP+HOG appearance features pooled over frames — always available.
    Research: Mauthner et al. (2009), Csurka et al. (2018), Arias & Millán (2018)."""
    _Z: FeatureDict = {
        **{f"app_lbp_mean_{i}":0.0 for i in range(10)},
        **{f"app_lbp_std_{i}": 0.0 for i in range(10)},
        "app_grad_energy_mean":0.0,"app_grad_energy_std":0.0,
        **{f"app_hog_mean_{i}":0.0 for i in range(6)},
        **{f"app_hog_std_{i}": 0.0 for i in range(6)},
    }
    if not gray_frames:
        return _Z
    T = (64, 64)
    def _lbp(g):
        off = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        code = np.zeros(g.shape, dtype=np.uint8)
        for b,(dy,dx) in enumerate(off):
            s = np.roll(np.roll(g,dy,0),dx,1)
            code |= ((g >= s).astype(np.uint8) << b)
        trans = np.zeros(g.shape, dtype=int)
        for b in range(8):
            trans += ((code>>b)&1)^((code>>((b+1)%8))&1)
        unif = trans <= 2
        bc   = sum(((code>>b)&1) for b in range(8))
        h    = np.zeros(10, dtype=float)
        for k in range(9): h[k] = np.sum((bc==k)&unif)
        h[9] = np.sum(~unif)
        return h / (h.sum()+1e-7)
    lh, ge, hh = [], [], []
    for frame in gray_frames:
        g   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        g64 = cv2.resize(g, T)
        lh.append(_lbp(g64))
        gx  = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 1, 0)
        gy  = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(np.abs(gy), np.abs(gx))
        ge.append(float(mag.mean()))
        h6, _ = np.histogram(ang.flatten(), bins=6, range=(0, np.pi), weights=mag.flatten())
        hh.append(h6 / (h6.sum()+1e-7))
    la = np.array(lh); ha = np.array(hh); ga = np.array(ge)
    res: FeatureDict = {
        "app_grad_energy_mean": float(ga.mean()),
        "app_grad_energy_std":  float(ga.std()) if len(ga) > 1 else 0.0,
    }
    for i in range(10):
        res[f"app_lbp_mean_{i}"] = float(la.mean(0)[i])
        res[f"app_lbp_std_{i}"]  = float(la.std(0)[i] if len(la) > 1 else 0.0)
    for i in range(6):
        res[f"app_hog_mean_{i}"] = float(ha.mean(0)[i])
        res[f"app_hog_std_{i}"]  = float(ha.std(0)[i] if len(ha) > 1 else 0.0)
    return res


def extract_video_metadata_features(n_frames: int, fps: float,
                                     duration_sec: float) -> FeatureDict:
    """3 temporal scalars appended to every group's feature vector (A–I).
    After normalisation these are contextual cues, not confounds.
    Research: Sargano et al. (2017) — use all available temporal cues."""
    return {
        "meta_num_frames":   float(n_frames),
        "meta_duration_sec": float(duration_sec),
        "meta_fps":          float(fps),
    }


def blend_trajectory_features(track_features: List[FeatureDict],
                               aggregator: "FeatureAggregator"
                               ) -> Tuple[FeatureDict, int, bool]:
    """Aggregate trajectory features; return (dict, n_tracks, indicator_present)."""
    if not track_features:
        return {}, 0, False
    return aggregator.aggregate(track_features) or {}, len(track_features), True
