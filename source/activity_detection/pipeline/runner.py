from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..detection.ground_truth import GroundTruthDetector
from ..detection.mog2 import MOG2Detector
from ..features.aggregation import FeatureAggregator
from ..features.optical_flow import OpticalFlowEstimator
from ..features.video_features import (
    extract_appearance_features,
    extract_diagnostic_features,
    extract_global_flow_features,
    extract_temporal_context,
)
from ..interfaces import (
    BBox,
    Detector,
    FailureReason,
    FeatureDict,
    FeatureExtractor,
    Preprocessor,
    Tracker,
    VideoProcessingResult,
)
from ..tracking.analysis import TrackerAnalyzer
from ..tracking.tfcr import TFCRTracker
from ..viz.explain import ExplainabilityVisualizer

log = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, preprocessor: Preprocessor, detector: Detector,
                 tracker: Tracker, optical_flow: Optional[OpticalFlowEstimator] = None,
                 feature_extractors: Optional[List[FeatureExtractor]] = None,
                 aggregator: Optional[FeatureAggregator] = None,
                 min_trajectory_len: int = 30,
                 use_dense_trajectories: bool = False,
                 enable_multifamily: bool = False,
                 filter_short_tracks: bool = False,
                 global_flow_step: int = 5,
                 appearance_frame_count: int = 30):
        self.preprocessor = preprocessor
        self.detector = detector
        self.tracker = tracker
        self.optical_flow = optical_flow
        self.feature_extractors = feature_extractors or []
        self.aggregator = aggregator or FeatureAggregator()
        self.min_trajectory_len = min_trajectory_len
        self.use_dense_trajectories = use_dense_trajectories
        self.enable_multifamily = enable_multifamily
        # False keeps the historical behaviour: every track with >= 2 points reaches
        # the trajectory features (wavelet/interaction apply their own minimum).
        # True drops tracks shorter than min_trajectory_len before feature extraction.
        self.filter_short_tracks = filter_short_tracks
        # Multi-family mode only: frame gap for the global optical-flow descriptor, and the
        # approximate number of evenly spaced frames used by the appearance descriptor.
        self.global_flow_step = max(1, int(global_flow_step))
        self.appearance_frame_count = max(1, int(appearance_frame_count))

    def process_video(self, video_path: Path,
                      collect_diagnostics: bool = False,
                      frame_sampling: int = 1,
                      visualise_path: Optional[Path] = None,
                      collect_frame_detections: bool = False) -> VideoProcessingResult:
        video_path = Path(video_path)
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                log.error("Cannot open video file: %s", video_path.name)
                return VideoProcessingResult(
                    tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                    n_frames_processed=0, failure=FailureReason(
                        file_missing=True,
                        error_message=f"cv2.VideoCapture could not open {video_path.name}"))
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            log.error("Failed to open video %s: %s\n%s", video_path.name, e, tb_str)
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=0, failure=FailureReason(
                    unknown_error=True, error_message=str(e), traceback=tb_str))

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Frame rate of the video, falling back to 10.0 when the container does not
        # report one. It is passed to every feature extractor as context["fps"] so
        # that speeds, accelerations and turning rates are in physical units
        # (px/s, px/s^2, rad/s) and comparable across videos.
        _cap_fps = float(cap.get(cv2.CAP_PROP_FPS))
        _cap_fps = _cap_fps if _cap_fps > 0 else 10.0

        _visualizer: Optional[ExplainabilityVisualizer] = None
        if visualise_path is not None:
            try:
                Path(visualise_path).parent.mkdir(parents=True, exist_ok=True)
                _visualizer = ExplainabilityVisualizer(
                    output_path=str(visualise_path),
                    fps=_cap_fps,
                    frame_size=(orig_w, orig_h),
                )
            except Exception as _ve:
                log.warning("ExplainabilityVisualizer init failed for %s: %s", video_path.name, _ve)
                _visualizer = None

        used_gt = False
        if isinstance(self.detector, GroundTruthDetector):
            used_gt = self.detector.load_for_video(video_path.name, orig_w, orig_h)
            if not used_gt and self.detector.fallback_to_mog2:
                print(f"  [GT] Fallback to MOG2 for {video_path.name}")
                self.detector = MOG2Detector(
                    history=20, var_threshold=16, detect_shadows=False,
                    morph_close_iter=2, morph_open_iter=1, min_contour_area=500
                )

        self.detector.reset()
        self.tracker.reset()
        self.preprocessor.reset()

        prev_gray = None
        flow_mags = []
        flow_angles = []
        diagnostics = []
        frame_no = 0
        n_detections = 0
        all_gray_frames = []

        detections_per_frame = []
        tracks_per_frame = []
        frame_detections: List[List[BBox]] = []


        try:
            while True:
                ret, raw = cap.read()
                if not ret:
                    break
                # Guard: cap.read() can return ret=True with a corrupt/empty frame.
                # cv2.resize (inside the preprocessor) will crash on zero-size arrays.
                if raw is None or raw.size == 0:
                    log.warning("process_video: empty frame at frame_no=%d — skipping", frame_no)
                    frame_no += 1
                    continue
                if frame_sampling > 1 and frame_no % frame_sampling != 0:
                    frame_no += 1
                    continue

                pre_result = self.preprocessor.process(raw)
                if collect_diagnostics and pre_result.diagnostics:
                    diagnostics.append(pre_result.diagnostics)

                dets = self.detector.detect(pre_result.gray)
                n_detections += len(dets)
                detections_per_frame.append(len(dets))

                # Trackers with an appearance model (TFCRTracker) need the current
                # grayscale frame; the geometric trackers (centroid, IoU+Hungarian,
                # SORT) only use the detections.
                if isinstance(self.tracker, TFCRTracker):
                    self.tracker.update(dets, frame_gray=pre_result.gray)
                else:
                    self.tracker.update(dets)
                active = self.tracker.get_active_tracks()
                tracks_per_frame.append(len(active))
                if collect_frame_detections:
                    frame_detections.append([t["bbox"] for t in active if "bbox" in t])
                # Provide track-quality feedback to the adaptive detector (1-frame lag).
                # This is a no-op for all non-adaptive detector types.
                if hasattr(self.detector, "feedback"):
                    self.detector.feedback(active)

                if self.optical_flow and prev_gray is not None:
                    flow_result = self.optical_flow.compute(prev_gray, pre_result.gray)
                    flow_mags.append(flow_result.mean_magnitude)
                    flow_angles.append(flow_result.mean_angle_deg)

                if _visualizer is not None:
                    _diag_now = pre_result.diagnostics if pre_result.diagnostics else {}
                    _dense_flow: Optional[np.ndarray] = None
                    # Expose dense flow if the detector computed it (DenseOpticalFlowDetector)
                    if prev_gray is not None and hasattr(self.detector, "_prev_gray"):
                        try:
                            _dense_flow = cv2.calcOpticalFlowFarneback(
                                prev_gray, pre_result.gray,
                                None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                        except Exception:
                            _dense_flow = None
                    _visualizer.write_frame(
                        frame_bgr=pre_result.bgr,
                        tracks=active,
                        diagnostics=_diag_now,
                        flow=_dense_flow,
                    )

                prev_gray = pre_result.gray
                frame_no += 1
                if self.use_dense_trajectories:
                    all_gray_frames.append(pre_result.gray)

            cap.release()
            if _visualizer is not None:
                _visualizer.release()
                _visualizer = None
        except Exception as e:
            cap.release()
            if _visualizer is not None:
                try:
                    _visualizer.release()
                except Exception:
                    pass
                _visualizer = None
            import traceback as _tb
            tb_str = _tb.format_exc()
            log.error(
                "Video %s failed during frame processing: %s\n%s",
                video_path.name, e, tb_str,
            )
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=frame_no,
                failure=FailureReason(unknown_error=True, error_message=str(e), traceback=tb_str))

        detection_stats = {
            "mean": float(np.mean(detections_per_frame)) if detections_per_frame else 0.0,
            "std": float(np.std(detections_per_frame)) if detections_per_frame else 0.0,
            "max": max(detections_per_frame) if detections_per_frame else 0,
            "zero_frames": sum(1 for d in detections_per_frame if d == 0),
            "total_frames": len(detections_per_frame),
        }
        tracking_stats = {
            "mean_tracks": float(np.mean(tracks_per_frame)) if tracks_per_frame else 0.0,
            "std_tracks": float(np.std(tracks_per_frame)) if tracks_per_frame else 0.0,
            "max_tracks": max(tracks_per_frame) if tracks_per_frame else 0,
            "total_frames": len(tracks_per_frame),
            "pruned_count": getattr(self.tracker, "pruned_count", 0),
        }

        if n_detections == 0 and frame_no > 0 and not self.enable_multifamily:
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=frame_no, used_gt=used_gt,
                detection_stats=detection_stats, tracking_stats=tracking_stats,
                failure=FailureReason(detection_failure=True),
                frame_detections=frame_detections, frame_size=(orig_w, orig_h))

        tracks = self.tracker.get_active_tracks()
        if self.filter_short_tracks:
            tracks = [t for t in tracks if len(t["trajectory"]) >= self.min_trajectory_len]
        mean_mag = float(np.mean(flow_mags)) if flow_mags else 0.0
        mean_ang = float(np.mean(flow_angles)) if flow_angles else 0.0
        tracker_metrics = TrackerAnalyzer.compute_tracker_metrics(tracks, frame_no)

        context = {"mean_flow_magnitude": mean_mag, "mean_flow_angle_deg": mean_ang,
                   "all_tracks": tracks,
                   # Sampled videos advance the tracker once per kept frame, so the
                   # time between trajectory points is frame_sampling / fps seconds.
                   "fps": _cap_fps / max(int(frame_sampling), 1)}
        if self.use_dense_trajectories and len(all_gray_frames) >= 2:
            mid = len(all_gray_frames) // 2
            context["prev_gray"] = all_gray_frames[max(0, mid-1)]
            context["curr_gray"] = all_gray_frames[mid]

        # Research: Mauthner et al. (2009); flow/appearance compensate when
        # tracks are absent in multi-family mode.
        track_feats:   List[FeatureDict] = []
        track_lengths: List[int]         = []

        for track in tracks:
            tlen = len(track["trajectory"])
            track_lengths.append(tlen)
            combined: FeatureDict = {}
            for extractor in self.feature_extractors:
                try:
                    fv = extractor.extract_track(track, **context)
                    if fv:
                        combined.update(fv)
                except Exception:
                    pass
            if combined:
                track_feats.append(combined)

        video_fv: Optional[FeatureDict] = self.aggregator.aggregate(track_feats) if track_feats else None

        if self.enable_multifamily:
            diag_feats   = extract_diagnostic_features(diagnostics)
            flow_feats   = extract_global_flow_features(all_gray_frames, flow_step=self.global_flow_step)
            step         = max(1, len(all_gray_frames) // self.appearance_frame_count)
            appear_feats = extract_appearance_features(all_gray_frames[::step])
            n_tj   = len(track_feats)
            tblock = video_fv or {}
            tblock["indicator_tracks_present"] = float(1 if n_tj > 0 else 0)
            tblock["meta_n_traj_tracks"]       = float(n_tj)
            temporal_feats = extract_temporal_context(tracks, total_frames=frame_no)
            blended: FeatureDict = {}
            blended.update(diag_feats)
            blended.update(flow_feats)
            blended.update(appear_feats)
            blended.update(tblock)
            blended.update(temporal_feats)
            blended.setdefault("n_valid_tracks", float(n_tj))
            video_fv = blended

        else:
            # Standard path: fail gracefully if no features produced
            if not track_feats:
                fail = (FailureReason(detection_failure=True) if not tracks
                        else FailureReason(feature_failure=True))
                return VideoProcessingResult(
                    tracks=tracks, mean_flow_magnitude=mean_mag,
                    mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                    used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                    track_lengths=track_lengths,
                    detection_stats=detection_stats, tracking_stats=tracking_stats,
                    failure=fail, tracker_metrics=tracker_metrics,
                    frame_detections=frame_detections, frame_size=(orig_w, orig_h))
            if video_fv is None:
                return VideoProcessingResult(
                    tracks=tracks, mean_flow_magnitude=mean_mag,
                    mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                    used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                    track_lengths=track_lengths,
                    detection_stats=detection_stats, tracking_stats=tracking_stats,
                    failure=FailureReason(feature_failure=True),
                    tracker_metrics=tracker_metrics,
                    frame_detections=frame_detections, frame_size=(orig_w, orig_h))
            temporal_feats = extract_temporal_context(tracks, total_frames=frame_no)
            video_fv.update(temporal_feats)

        feat_count = len(video_fv) if video_fv else 0

        return VideoProcessingResult(
            tracks=tracks,
            mean_flow_magnitude=mean_mag,
            mean_flow_angle_deg=mean_ang,
            n_frames_processed=frame_no,
            used_gt=used_gt,
            preprocessing_diagnostics=diagnostics,
            track_features=track_feats,
            video_feature=video_fv,
            track_lengths=track_lengths,
            feature_count=feat_count,
            tracker_metrics=tracker_metrics,
            detection_stats=detection_stats,
            tracking_stats=tracking_stats,
            frame_detections=frame_detections,
            frame_size=(orig_w, orig_h),
        )
