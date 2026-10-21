"""Shared track state and lifecycle for the trackers.

Every tracker keeps a dict of live tracks, an id counter and a count of unmatched frames per track;
this base class owns that once and a subclass only supplies the matching step.

A track that stays unmatched for more than ``max_disappeared`` frames is retired. By default it is
archived in ``finished`` and returned by ``get_all_tracks()`` together with the live tracks, because a
person who walks through the scene and leaves is a real observation. ``get_active_tracks()`` returns
only the tracks alive now. With ``keep_finished_tracks=False`` retired tracks are dropped, so only the
tracks alive at the last frame reach feature extraction (the old behaviour).
"""
from __future__ import annotations

import abc
from typing import Dict, List

from ..interfaces import BBox, Track, Tracker


class BaseTracker(Tracker, abc.ABC):
    """Holds the track state; a subclass implements only the matching step.

    Attributes:
        tracks: live tracks by id (each has ``id``, ``centroid``, ``trajectory``, ``frames``, ``disappeared``).
        finished: retired tracks (only filled when ``keep_finished_tracks`` is true).
        pruned_count: number of tracks retired so far.
        frame_idx: index of the current frame (-1 before the first update).
    """

    #: Label stored on every track; defaults to the class name.
    tracker_type: str | None = None

    def __init__(
        self,
        max_disappeared: int = 30,
        max_trajectory_len: int = 300,
        keep_finished_tracks: bool = True,
    ) -> None:
        """Create an empty tracker.

        Args:
            max_disappeared: unmatched frames in a row after which a track is retired.
            max_trajectory_len: number of most recent centroids kept per track.
            keep_finished_tracks: archive retired tracks so they still reach feature extraction.
        """
        self.max_disappeared = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.keep_finished_tracks = keep_finished_tracks
        # This is the id that the next new track gets; it counts up and starts again at
        # 0 for every video.
        self.next_id: int = 0
        # This is where the live tracks are kept, with the track id as the key.
        self.tracks: Dict[int, Track] = {}
        # This is where tracks go once they are retired, so that they still reach
        # feature extraction.
        self.finished: List[Track] = []
        # This is how many tracks have been retired so far, whether or not they were
        # kept.
        self.pruned_count: int = 0
        # This is the index of the frame being processed; it is -1 until the first
        # update.
        self.frame_idx: int = -1          # index of the frame being processed

    @abc.abstractmethod
    def _match_and_update(self, detections: List[BBox]) -> None:
        """Match *detections* to existing tracks and register new ones.

        Implementations must call ``_new_track`` / ``_update_track`` and
        increment ``disappeared`` counters on unmatched existing tracks.
        They must NOT call ``_prune_expired`` — the base ``update`` does that.
        """

    def _begin_frame(self, detections: List[BBox]) -> None:
        """Hook run at the start of every ``update`` (e.g. a Kalman predict step)."""

    def _new_track(self, centroid, bbox: BBox | None = None, **extra) -> None:
        """Register a new live track starting at *centroid* in the current frame.

        Extra keyword arguments are stored on the track (for example a Kalman
        filter). Returns nothing; the track is available as ``self.tracks[id]``.
        """
        track: Track = {
            "id": self.next_id,
            "centroid": centroid,
            "trajectory": [centroid],
            "frames": [self.frame_idx],   # frame index of each trajectory point
            "disappeared": 0,
            "tracker_type": self.tracker_type or self.__class__.__name__,
            **extra,
        }
        if bbox is not None:
            track["bbox"] = bbox
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track: Track, centroid, bbox: BBox | None = None) -> None:
        """Append a matched observation to *track* and reset its disappearance counter."""
        track["centroid"] = centroid
        track["disappeared"] = 0
        track["trajectory"].append(centroid)
        track.setdefault("frames", []).append(self.frame_idx)
        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len :]
            track["frames"] = track["frames"][-self.max_trajectory_len :]
        if bbox is not None:
            track["bbox"] = bbox

    def _increment_disappeared(self, track_ids: List[int]) -> None:
        """Count one more unmatched frame for each track in *track_ids*."""
        for tid in track_ids:
            self.tracks[tid]["disappeared"] += 1

    def _prune_expired(self) -> None:
        """Retire tracks unmatched for more than ``max_disappeared`` frames.

        Retired tracks are moved to ``self.finished`` when ``keep_finished_tracks``
        is true, otherwise they are dropped (legacy behaviour).
        """
        expired = [
            tid
            for tid, t in self.tracks.items()
            if t["disappeared"] > self.max_disappeared
        ]
        for tid in expired:
            track = self.tracks.pop(tid)
            if self.keep_finished_tracks:
                self.finished.append(track)
            self.pruned_count += 1

    def _is_reportable(self, track: Track) -> bool:
        """Whether a track may be reported to feature extraction (SORT overrides this)."""
        return True

    def update(self, detections: List[BBox]) -> List[Track]:
        """Advance one frame: match *detections*, retire stale tracks, return live tracks."""
        self.frame_idx += 1
        self._begin_frame(detections)
        if not detections:
            self._increment_disappeared(list(self.tracks))
        else:
            self._match_and_update(detections)
        self._prune_expired()
        return list(self.tracks.values())

    def get_active_tracks(self) -> List[Track]:
        """Tracks that are alive right now (matched within ``max_disappeared`` frames)."""
        return list(self.tracks.values())

    def get_all_tracks(self) -> List[Track]:
        """Every reportable track of the video: retired ones first, then live ones.

        This is the list that feature extraction should use. With
        ``keep_finished_tracks=False`` it equals ``get_active_tracks()``.
        """
        retired = [t for t in self.finished if self._is_reportable(t)]
        return retired + self.get_active_tracks()

    def reset(self) -> None:
        """Forget all tracks; call before every new video."""
        self.tracks.clear()
        self.finished = []
        self.next_id = 0
        self.pruned_count = 0
        self.frame_idx = -1
