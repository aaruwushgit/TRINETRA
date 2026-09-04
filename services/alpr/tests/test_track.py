"""Multi-frame tracking."""

from __future__ import annotations

import pytest

from alpr.detect import Detection
from alpr.track import Tracker


def box(x, y, w=0.10, h=0.04, confidence=0.9) -> Detection:
    return Detection(x, y, x + w, y + h, confidence)


def drive(tracker, frames):
    """Feed a list of per-frame detection lists."""
    for index, detections in enumerate(frames):
        tracker.update(detections, index)
    return tracker


class TestConfiguration:
    @pytest.mark.parametrize(
        "kwargs",
        [{"iou_threshold": 0}, {"iou_threshold": 1.5}, {"max_age": -1}, {"min_hits": 0}],
    )
    def test_rejects_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            Tracker(**kwargs)


class TestAssociation:
    def test_one_vehicle_keeps_one_id(self):
        tracker = Tracker(min_hits=1)
        # A plate drifting across the frame, as a car passing would.
        frames = [[box(0.10 + i * 0.01, 0.5)] for i in range(10)]
        drive(tracker, frames)
        assert len({t.track_id for t in tracker.tracks}) == 1
        assert tracker.tracks[0].hits == 10

    def test_two_vehicles_get_two_ids(self):
        tracker = Tracker(min_hits=1)
        frames = [[box(0.1 + i * 0.01, 0.2), box(0.7 - i * 0.01, 0.8)] for i in range(6)]
        drive(tracker, frames)
        assert len(tracker.tracks) == 2

    def test_a_jump_too_far_starts_a_new_track(self):
        # No overlap means it cannot be the same plate.
        tracker = Tracker(min_hits=1)
        drive(tracker, [[box(0.1, 0.1)], [box(0.8, 0.8)]])
        assert len(tracker.tracks) == 2

    def test_each_detection_claims_one_track(self):
        # Two overlapping detections must not both update the same track.
        tracker = Tracker(min_hits=1)
        tracker.update([box(0.30, 0.5)], 0)
        tracker.update([box(0.31, 0.5), box(0.32, 0.5)], 1)
        assert len(tracker.tracks) == 2


class TestMinHits:
    def test_single_frame_detection_is_never_confirmed(self):
        # The false-positive filter. The detector's errors are overwhelmingly
        # single-frame spurious boxes; a real plate persists.
        tracker = Tracker(min_hits=3, max_age=1)
        drive(tracker, [[box(0.5, 0.5)], [], [], []])
        assert tracker.finish() == []

    def test_a_persistent_plate_is_confirmed(self):
        tracker = Tracker(min_hits=3)
        confirmed = drive(tracker, [[box(0.10 + i * 0.005, 0.5)] for i in range(5)]).tracks
        assert confirmed[0].confirmed is True

    def test_confirmation_needs_exactly_min_hits(self):
        tracker = Tracker(min_hits=3)
        for frame in range(2):
            tracker.update([box(0.10 + frame * 0.005, 0.5)], frame)
        assert not any(t.confirmed for t in tracker.tracks)
        tracker.update([box(0.12, 0.5)], 2)
        assert any(t.confirmed for t in tracker.tracks)


class TestMaxAge:
    def test_survives_a_brief_disappearance(self):
        # Plates vanish behind wipers and sign posts; killing the track on the
        # first missed frame would log one vehicle twice.
        tracker = Tracker(min_hits=1, max_age=5)
        tracker.update([box(0.30, 0.5)], 0)
        for frame in range(1, 4):
            tracker.update([], frame)
        tracker.update([box(0.31, 0.5)], 4)
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].hits == 2

    def test_dies_after_max_age(self):
        tracker = Tracker(min_hits=1, max_age=2)
        tracker.update([box(0.3, 0.5)], 0)
        for frame in range(1, 6):
            tracker.update([], frame)
        assert tracker.tracks == []
        assert len(tracker.finished) == 1

    def test_a_returning_vehicle_is_a_new_track(self):
        # Correct: after a long gap this is a separate pass, and cross-pass
        # duplicates are alpr.dedup's job, not the tracker's.
        tracker = Tracker(min_hits=1, max_age=2)
        tracker.update([box(0.3, 0.5)], 0)
        for frame in range(1, 6):
            tracker.update([], frame)
        tracker.update([box(0.3, 0.5)], 6)
        assert len(tracker.finished) == 1
        assert tracker.tracks[0].track_id != tracker.finished[0].track_id


class TestEmission:
    def test_one_emission_per_vehicle(self):
        # The Phase 6 exit criterion: emitted events equal vehicle count.
        tracker = Tracker(min_hits=2, max_age=2)
        for vehicle in range(3):
            base = 0.1 + vehicle * 0.3
            for step in range(5):
                tracker.update([box(base + step * 0.005, 0.5)], vehicle * 10 + step)
            for gap in range(4):
                tracker.update([], vehicle * 10 + 5 + gap)
        tracker.finish()
        assert len(list(tracker.completed())) == 3

    def test_completed_does_not_repeat(self):
        tracker = Tracker(min_hits=1, max_age=1)
        tracker.update([box(0.3, 0.5)], 0)
        tracker.finish()
        assert len(list(tracker.completed())) == 1
        assert list(tracker.completed()) == []

    def test_finish_retires_live_tracks(self):
        # A vehicle still in frame on the last frame must still be logged.
        tracker = Tracker(min_hits=1)
        tracker.update([box(0.3, 0.5)], 0)
        assert tracker.tracks
        assert len(tracker.finish()) == 1
        assert tracker.tracks == []

    def test_unconfirmed_tracks_are_never_emitted(self):
        tracker = Tracker(min_hits=5)
        tracker.update([box(0.3, 0.5)], 0)
        assert tracker.finish() == []


class TestTrack:
    def test_best_returns_the_clearest_sighting(self):
        # The crop worth reading is the most confident one, not the last.
        tracker = Tracker(min_hits=1)
        tracker.update([box(0.30, 0.5, confidence=0.4)], 0)
        tracker.update([box(0.31, 0.5, confidence=0.95)], 1)
        tracker.update([box(0.32, 0.5, confidence=0.6)], 2)
        assert tracker.tracks[0].best.confidence == 0.95

    def test_duration_and_history(self):
        tracker = Tracker(min_hits=1)
        for frame in range(4):
            tracker.update([box(0.30 + frame * 0.005, 0.5)], frame)
        track = tracker.tracks[0]
        assert track.duration == 4
        assert len(track.history) == 4
