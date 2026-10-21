"""Abstract interfaces and shared data containers of the pipeline.

Every stage is an abstract class registered by name in ``registry``, so a YAML config can swap
implementations. Boxes, tracks and feature vectors are plain dicts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Plain-dict aliases keep the stages loosely coupled and trivially serialisable.
BBox        = Dict[str, Any]     # {x, y, w, h, cx, cy, centroid, area} - see detection.utils.make_bbox
Track       = Dict[str, Any]     # {id, centroid, trajectory, frames, disappeared, ...} - see tracking.base
FeatureDict = Dict[str, float]   # feature name -> value


@dataclass
class PreprocessResult:
    """Output of a preprocessor for one frame.

    Attributes:
        bgr: the resized colour frame (not enhanced).
        gray: the grey frame given to the detector (enhanced when the preprocessor is distortion aware).
        diagnostics: per-frame quality measurements (sharpness, brightness, ...).
        enhanced: whether any enhancement was applied.
    """

    bgr: np.ndarray
    gray: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    enhanced: bool = False


@dataclass
class OpticalFlowResult:
    """Sparse Lucas-Kanade optical flow summary between two frames (pixels per frame)."""

    mean_magnitude: float
    mean_angle_deg: float
    flow_vectors: Optional[np.ndarray] = None
    prev_corners: Optional[np.ndarray] = None


@dataclass
class FailureReason:
    """Why a video produced no feature vector (exactly one flag is normally set)."""

    file_missing: bool = False
    detection_failure: bool = False
    tracking_failure: bool = False
    track_too_short: bool = False
    feature_failure: bool = False
    unknown_error: bool = False
    low_detection_density: bool = False
    track_pruned_often: bool = False
    error_message: str = ""
    traceback: str = ""


@dataclass
class VideoProcessingResult:
    """Everything the pipeline learned from one video.

    ``tracks`` are the tracks that reached feature extraction. Three yields are kept apart:
    ``video_feature`` is not ``None`` (a feature vector exists), ``track_lengths`` is not empty
    (at least one track), and ``n_usable_tracks > 0`` (a track with at least ``min_trajectory_len`` points).
    """

    tracks: List[Track]
    mean_flow_magnitude: float
    mean_flow_angle_deg: float
    n_frames_processed: int
    used_gt: bool = False
    n_usable_tracks: int = 0     # tracks with >= min_trajectory_len points
    preprocessing_diagnostics: List[dict] = field(default_factory=list)
    track_features: List[FeatureDict] = field(default_factory=list)
    video_feature: Optional[FeatureDict] = None
    track_lengths: List[int] = field(default_factory=list)
    feature_count: int = 0
    failure: Optional[FailureReason] = None
    tracker_metrics: Optional[Dict[str, Any]] = None
    detection_stats: Dict[str, Any] = field(default_factory=dict)
    tracking_stats: Dict[str, Any] = field(default_factory=dict)
    # Per-frame active-track bounding boxes, one list per processed frame.
    # Only populated when `collect_frame_detections=True` is passed to
    # `PipelineRunner.process_video` (off by default, it holds one bbox
    # list per frame in memory, which isn't needed for normal runs and is
    # only worth the cost when actually building a FiftyOne dataset).
    frame_detections: List[List[BBox]] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None  # (width, height)


class Detector(abc.ABC):
    """Turns one grey frame into boxes of moving objects.

    A detector keeps a background model, so it handles one video at a time and is ``reset()`` between videos.
    """

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Return the boxes found in *frame* and update the internal background model."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Discard the background model before a new video."""


class Tracker(abc.ABC):
    """Associates per-frame detections over time into tracks (trajectories).

    A track is a dict with at least ``id``, ``centroid``, ``trajectory`` (list of
    (x, y) centroids), ``frames`` (frame index of each centroid) and ``disappeared``.
    """

    @abc.abstractmethod
    def update(self, detections: List[BBox]) -> List[Track]:
        """Consume one frame of detections and return the tracks alive afterwards."""

    @abc.abstractmethod
    def get_active_tracks(self) -> List[Track]:
        """Tracks that are alive right now."""

    def get_all_tracks(self) -> List[Track]:
        """All tracks of the current video, including ones already retired.

        Trackers that retire tracks should override this; the default returns only
        the live tracks, which is the legacy behaviour.
        """
        return self.get_active_tracks()

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget all state before a new video."""


class Preprocessor(abc.ABC):
    """Converts a raw BGR frame into the greyscale image the detector sees.

    Besides the greyscale image, a preprocessor reports per-frame quality
    ``diagnostics`` (see ``preprocessing.frame_quality_metrics``) that are pooled
    into the video-level image-quality features.
    """

    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Resize / enhance *frame* and return the images plus quality diagnostics."""

    def reset(self) -> None:
        """Clear any per-video state (most preprocessors have none)."""


class FeatureExtractor(abc.ABC):
    """Computes scalar descriptors from one track.

    Per-track descriptors are pooled over a video's tracks (mean / max / std) by
    ``features.aggregation.FeatureAggregator`` into the video-level feature vector.
    """

    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        """Names of the descriptors this extractor can return."""

    @abc.abstractmethod
    def extract_track(self, track: Track, **context: Any) -> Optional[FeatureDict]:
        """Descriptors of *track*, or ``None`` when the track is too short for them.

        ``context`` carries video-level values: ``fps`` (frames per second of the
        processed stream), ``all_tracks`` (every track of the video, for interaction
        descriptors) and the video-level Lucas-Kanade flow summary.
        """


class Model(abc.ABC):
    """Minimal classifier contract used by the evaluation code (scikit-learn style)."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """Train on feature matrix *X* and integer labels *y*; return ``self``."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted integer label for every row of *X*."""

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Class probabilities, or ``None`` when the model cannot provide them."""
        return None

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Per-feature importance, or ``None`` when the model has no such notion."""
        return None

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Class labels in the column order of ``predict_proba``."""
        return None
