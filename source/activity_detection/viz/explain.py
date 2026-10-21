"""Annotated explainability video: tracks, flow arrows and a diagnostics sidebar."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


class ExplainabilityVisualizer:
    """
    Renders an annotated diagnostic video overlay showing optical flow arrows,
    bounding boxes, trajectory tails, and a live feature dashboard sidebar.
    Uses only OpenCV drawing (no Matplotlib) for real-time performance.
    """

    def __init__(self, output_path: str, fps: float = 10.0, frame_size: tuple = (640, 480)):
        # This is the width of the panel at the side of the frame that shows the track
        # statistics.
        self.dashboard_width = 300
        self.out_w = frame_size[0] + self.dashboard_width
        self.out_h = frame_size[1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # This writes the annotated frames to the output video.
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (self.out_w, self.out_h))
        rng = np.random.default_rng(42)
        # This is a fixed set of colours, one per track id, so a track keeps its colour.
        self.colors = rng.integers(60, 255, (200, 3), dtype=np.uint8)

    def draw_optical_flow(self, frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
        """Draws Farnebaeck dense optical flow as a sparse grid of arrows."""
        h, w = frame.shape[:2]
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        ys = ys.ravel().astype(int)
        xs = xs.ravel().astype(int)
        # clamp to valid range after any resize mismatch
        ys = np.clip(ys, 0, flow.shape[0] - 1)
        xs = np.clip(xs, 0, flow.shape[1] - 1)
        fx = flow[ys, xs, 0]
        fy = flow[ys, xs, 1]
        mag = np.hypot(fx, fy)
        mask = mag > 1.5
        vis = frame.copy()
        for x1, y1, dvx, dvy in zip(xs[mask], ys[mask], fx[mask], fy[mask]):
            x2, y2 = int(x1 + dvx), int(y1 + dvy)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 230, 230), 1, tipLength=0.35)
        return vis

    def overlay_tracks(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Copy of *frame* with each track's box and trajectory tail drawn in a per-track colour."""
        vis = frame.copy()
        for track in tracks:
            tid = track.get("id", 0)
            color = self.colors[tid % len(self.colors)].tolist()

            bbox = track.get("bbox")
            if bbox:
                x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), \
                              int(bbox.get("w", 0)), int(bbox.get("h", 0))
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x, max(y - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            traj = track.get("trajectory", [])
            if len(traj) > 1:
                pts = np.array(traj, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2)
            if traj:
                cx, cy = int(traj[-1][0]), int(traj[-1][1])
                cv2.circle(vis, (cx, cy), 5, color, -1)

        return vis

    def build_dashboard(self, tracks: List[dict], diagnostics: dict) -> np.ndarray:
        """Side panel image listing the current tracks and the frame's quality diagnostics."""
        dash = np.zeros((self.out_h, self.dashboard_width, 3), dtype=np.uint8)
        W = self.dashboard_width

        def put(text, y, color=(200, 200, 200), scale=0.4, thickness=1):
            cv2.putText(dash, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)

        y = 24
        put("EXPLAINABILITY", y, color=(255, 255, 255), scale=0.55, thickness=2)
        y += 4
        cv2.line(dash, (6, y), (W - 6, y), (180, 180, 180), 1)
        y += 18

        put("[Quality Prior]", y, color=(80, 220, 80), scale=0.42, thickness=1)
        y += 18
        for k, v in diagnostics.items():
            label = k[:18]
            val = f"{v:.2f}" if isinstance(v, float) else str(v)[:10]
            put(f"  {label}: {val}", y)
            y += 16
            if y > self.out_h - 80:
                break

        y += 8
        cv2.line(dash, (6, y), (W - 6, y), (100, 100, 100), 1)
        y += 14

        put("[Active Track]", y, color=(80, 220, 220), scale=0.42, thickness=1)
        y += 18

        if tracks:
            longest = max(tracks, key=lambda t: len(t.get("trajectory", [])))
            tid = longest.get("id", "?")
            traj = longest.get("trajectory", [])
            n_pts = len(traj)

            put(f"  Track ID : {tid}", y)
            y += 15
            put(f"  Path pts : {n_pts}", y)
            y += 15

            # Instantaneous speed (px/frame) between last two points
            if n_pts >= 2:
                p1, p2 = traj[-2], traj[-1]
                speed = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                put(f"  Speed    : {speed:.1f} px/fr", y)
                y += 15

            # Trajectory bounding box displacement
            if n_pts >= 2:
                xs_t = [p[0] for p in traj]
                ys_t = [p[1] for p in traj]
                disp = float(np.hypot(xs_t[-1] - xs_t[0], ys_t[-1] - ys_t[0]))
                put(f"  Displace : {disp:.1f} px", y)
                y += 15

        y += 6
        put(f"Tracks visible: {len(tracks)}", y, color=(255, 200, 80), scale=0.42)

        return dash

    def write_frame(self, frame_bgr: np.ndarray, tracks: List[dict],
                    diagnostics: dict, flow: Optional[np.ndarray] = None):
        """Draw the tracks, optional flow arrows and the dashboard on the frame and append it to the video."""
        vis = frame_bgr.copy()

        # Resize vis to match declared output dimensions (guard against mismatch)
        target_h, target_w = self.out_h, self.out_w - self.dashboard_width
        if vis.shape[0] != target_h or vis.shape[1] != target_w:
            vis = cv2.resize(vis, (target_w, target_h))

        if flow is not None:
            vis = self.draw_optical_flow(vis, flow)

        vis = self.overlay_tracks(vis, tracks)
        dashboard = self.build_dashboard(tracks, diagnostics)
        final = np.hstack((vis, dashboard))
        self.writer.write(final)

    def release(self):
        self.writer.release()
