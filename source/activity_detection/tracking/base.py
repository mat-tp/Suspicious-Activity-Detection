"""Shared tracker state and lifecycle management.

All three concrete trackers (Centroid, IoU+Hungarian, SORT) previously
duplicated next_id, tracks, pruned_count, _prune_expired, get_active_tracks,
and reset.  This base class owns that state once.
"""
from __future__ import annotations

import abc
from typing import Dict, List

from ..interfaces import BBox, Track, Tracker


class BaseTracker(Tracker, abc.ABC):
    """Manages shared track state; subclasses implement only the matching step."""

    #: Label stored on every track; defaults to the class name.
    tracker_type: str | None = None

    def __init__(
        self,
        max_disappeared: int = 30,
        max_trajectory_len: int = 300,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.next_id: int = 0
        self.tracks: Dict[int, Track] = {}
        self.pruned_count: int = 0

    # ── Subclass contract ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _match_and_update(self, detections: List[BBox]) -> None:
        """Match *detections* to existing tracks and register new ones.

        Implementations must call ``_new_track`` / ``_update_track`` and
        increment ``disappeared`` counters on unmatched existing tracks.
        They must NOT call ``_prune_expired`` — the base ``update`` does that.
        """

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _begin_frame(self, detections: List[BBox]) -> None:
        """Hook run at the start of every ``update`` (e.g. a Kalman predict step)."""

    def _new_track(self, centroid, bbox: BBox | None = None, **extra) -> None:
        track: Track = {
            "id": self.next_id,
            "centroid": centroid,
            "trajectory": [centroid],
            "disappeared": 0,
            "tracker_type": self.tracker_type or self.__class__.__name__,
            **extra,
        }
        if bbox is not None:
            track["bbox"] = bbox
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track: Track, centroid, bbox: BBox | None = None) -> None:
        track["centroid"] = centroid
        track["disappeared"] = 0
        track["trajectory"].append(centroid)
        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len :]
        if bbox is not None:
            track["bbox"] = bbox

    def _increment_disappeared(self, track_ids: List[int]) -> None:
        for tid in track_ids:
            self.tracks[tid]["disappeared"] += 1

    def _prune_expired(self) -> None:
        expired = [
            tid
            for tid, t in self.tracks.items()
            if t["disappeared"] > self.max_disappeared
        ]
        for tid in expired:
            del self.tracks[tid]
            self.pruned_count += 1

    # ── Tracker interface ────────────────────────────────────────────────────

    def update(self, detections: List[BBox]) -> List[Track]:
        self._begin_frame(detections)
        if not detections:
            self._increment_disappeared(list(self.tracks))
        else:
            self._match_and_update(detections)
        self._prune_expired()
        return list(self.tracks.values())

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 0
        self.pruned_count = 0
