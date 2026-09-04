"""Cross-pass deduplication."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from alpr.dedup import Deduplicator

T0 = datetime(2026, 8, 9, 12, 0, 0)


def test_first_sighting_is_always_accepted():
    assert Deduplicator().accept("MH12AB1234", T0) is True


def test_repeat_within_cooldown_is_suppressed():
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    assert dedup.accept("MH12AB1234", T0 + timedelta(minutes=1)) is False


def test_repeat_after_cooldown_is_accepted():
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    assert dedup.accept("MH12AB1234", T0 + timedelta(minutes=6)) is True


def test_different_plates_do_not_interfere():
    dedup = Deduplicator(timedelta(minutes=5))
    assert dedup.accept("MH12AB1234", T0) is True
    assert dedup.accept("DAXY123", T0) is True


def test_cooldown_restarts_on_each_suppressed_sighting():
    # A car idling in view for 20 minutes should produce one row, not a new
    # one every time the window happens to expire.
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    for minute in range(1, 20):
        assert dedup.accept("MH12AB1234", T0 + timedelta(minutes=minute)) is False


def test_a_gap_longer_than_the_window_ends_suppression():
    # The vehicle left and came back — a genuinely new visit.
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    dedup.accept("MH12AB1234", T0 + timedelta(minutes=2))
    assert dedup.accept("MH12AB1234", T0 + timedelta(minutes=10)) is True


def test_zero_cooldown_accepts_everything():
    dedup = Deduplicator(timedelta(0))
    assert dedup.accept("X", T0) is True
    assert dedup.accept("X", T0) is True


def test_counts_suppressions():
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    for minute in range(1, 4):
        dedup.accept("MH12AB1234", T0 + timedelta(minutes=minute))
    assert dedup.suppressed_count == 3


def test_history_records_first_and_last_sighting():
    dedup = Deduplicator(timedelta(minutes=5))
    dedup.accept("MH12AB1234", T0)
    dedup.accept("MH12AB1234", T0 + timedelta(minutes=2))
    history = dedup.history("MH12AB1234")
    assert history.first_seen == T0
    assert history.last_seen == T0 + timedelta(minutes=2)
    assert history.count == 2


def test_memory_is_bounded():
    # A long run must not grow without limit.
    dedup = Deduplicator(timedelta(minutes=5), max_tracked=10)
    for i in range(100):
        dedup.accept(f"PLATE{i}", T0 + timedelta(seconds=i))
    assert dedup.tracked() == 10


def test_eviction_is_oldest_first():
    dedup = Deduplicator(timedelta(minutes=5), max_tracked=2)
    dedup.accept("A", T0)
    dedup.accept("B", T0 + timedelta(seconds=1))
    dedup.accept("C", T0 + timedelta(seconds=2))
    assert dedup.history("A") is None
    assert dedup.history("C") is not None


def test_reset_clears_state():
    dedup = Deduplicator()
    dedup.accept("A", T0)
    dedup.reset()
    assert dedup.tracked() == 0
    assert dedup.accept("A", T0) is True


@pytest.mark.parametrize(
    ("cooldown", "max_tracked"),
    [(timedelta(seconds=-1), 10), (timedelta(minutes=1), 0)],
)
def test_rejects_invalid_configuration(cooldown, max_tracked):
    with pytest.raises(ValueError):
        Deduplicator(cooldown, max_tracked=max_tracked)
