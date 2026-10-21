"""Tracker contract tests: BaseTracker lifecycle shared by all three trackers."""
import pytest

from activity_detection.detection.utils import make_bbox
from activity_detection.tracking.centroid import CentroidTracker
from activity_detection.tracking.iou_hungarian import IoUHungarianTracker
from activity_detection.tracking.sort import SORTTracker
from activity_detection.tracking.base import BaseTracker

ALL = [CentroidTracker, IoUHungarianTracker, SORTTracker]


@pytest.mark.parametrize("cls", ALL)
def test_all_trackers_share_base(cls):
    assert issubclass(cls, BaseTracker)


@pytest.mark.parametrize("cls", ALL)
def test_prune_and_reset(cls):
    t = cls(max_disappeared=2)
    t.update([make_bbox(10, 10, 40, 40)])
    for _ in range(4):
        t.update([])
    assert not t.tracks and t.pruned_count == 1
    t.reset()
    assert t.next_id == 0 and t.pruned_count == 0


def test_sort_only_reports_confirmed_tracks_and_keeps_its_label():
    t = SORTTracker(min_hits=3)
    t.update([make_bbox(10, 10, 40, 40)])
    assert t.get_active_tracks() == []              # hits=1 < min_hits
    t.tracks[0]["hits"] = 3
    assert [x["id"] for x in t.get_active_tracks()] == [0]
    assert t.tracks[0]["tracker_type"] == "sort"


def test_sort_keeps_a_slowly_moving_object_as_one_track():
    t = SORTTracker(min_hits=1)
    for i in range(10):
        t.update([make_bbox(100 + i, 100, 40, 40)])
    assert t.next_id == 1


def test_sort_kalman_ages_on_empty_frames():
    t = SORTTracker()
    t.update([make_bbox(10, 10, 40, 40)])
    t.update([])
    assert t.tracks[0]["kalman_tracker"].time_since_update == 1
    t.reset()
    assert t.frame_count == 0 and t._kalman_id_counter == 0


def _walk_through_and_leave(tracker, n_present=8, n_absent=6):
    """An object visible for n_present frames, then no detections at all."""
    for i in range(n_present):
        tracker.update([make_bbox(10 + i, 10, 20, 40)])
    for _ in range(n_absent):
        tracker.update([])


def test_retired_track_reaches_feature_extraction_by_default():
    t = IoUHungarianTracker(max_disappeared=3)
    _walk_through_and_leave(t)
    assert t.get_active_tracks() == []                 # the track was retired ...
    assert len(t.get_all_tracks()) == 1                # ... but is still reported
    assert len(t.get_all_tracks()[0]["trajectory"]) == 8
    assert t.pruned_count == 1


def test_legacy_mode_discards_retired_tracks():
    t = IoUHungarianTracker(max_disappeared=3, keep_finished_tracks=False)
    _walk_through_and_leave(t)
    assert t.get_all_tracks() == []
    assert t.pruned_count == 1


def test_reset_clears_the_archive():
    t = IoUHungarianTracker(max_disappeared=3)
    _walk_through_and_leave(t)
    t.reset()
    assert t.get_all_tracks() == [] and t.pruned_count == 0


def test_sort_does_not_report_unconfirmed_retired_tracks():
    t = SORTTracker(max_disappeared=2, min_hits=3)
    t.update([make_bbox(10, 10, 20, 40)])              # a single noise blob
    for _ in range(5):
        t.update([])
    assert t.get_all_tracks() == []
