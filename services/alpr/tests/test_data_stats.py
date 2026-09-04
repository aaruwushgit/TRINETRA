"""Dataset statistics and the Phase 1 exit criteria."""

from __future__ import annotations

from alpr.data import (
    ImageRecord,
    PlateBox,
    Region,
    Split,
    check_exit_criteria,
    compute_stats,
    split_records,
)
from alpr.data.stats import MIN_READABLE_PLATE_PX


def test_empty_dataset_does_not_divide_by_zero():
    stats = compute_stats([])
    assert stats.images == 0
    assert stats.plates == 0
    assert stats.plates_per_image == 0.0
    assert stats.median_plate_px == (0.0, 0.0)
    stats.report()  # must not raise


def test_counts_images_plates_and_groups():
    records = [
        ImageRecord(
            "clip1_frame_001",
            1000,
            1000,
            boxes=(PlateBox(0.3, 0.5, 0.1, 0.05), PlateBox(0.7, 0.5, 0.1, 0.05)),
        ),
        ImageRecord("clip1_frame_002", 1000, 1000, boxes=(PlateBox(0.5, 0.5, 0.1, 0.05),)),
        ImageRecord("clip2_frame_001", 1000, 1000),
    ]
    stats = compute_stats(records)
    assert stats.images == 3
    assert stats.plates == 3
    assert stats.groups == 2
    assert stats.images_without_plates == 1
    assert stats.plates_per_image == 1.0


def test_region_breakdown():
    records = [
        ImageRecord(
            "a",
            1000,
            1000,
            boxes=(
                PlateBox(0.3, 0.5, 0.1, 0.05, region=Region.INDIA),
                PlateBox(0.7, 0.5, 0.1, 0.05, region=Region.GERMANY),
                PlateBox(0.5, 0.2, 0.1, 0.05),
            ),
        )
    ]
    stats = compute_stats(records)
    assert stats.by_region == {"IN": 1, "DE": 1, "XX": 1}


def test_counts_plates_too_small_to_read():
    # 0.01 * 1000 = 10px wide: below the OCR floor, so Phase 4 cannot read it
    # no matter how well the detector localizes it.
    records = [
        ImageRecord("a", 1000, 1000, boxes=(PlateBox(0.5, 0.5, 0.01, 0.005),)),
        ImageRecord("b", 1000, 1000, boxes=(PlateBox(0.5, 0.5, 0.20, 0.100),)),
    ]
    stats = compute_stats(records)
    assert stats.tiny_plates == 1
    assert MIN_READABLE_PLATE_PX == 32


def test_counts_boxes_with_ocr_text():
    records = [
        ImageRecord(
            "a",
            100,
            100,
            boxes=(PlateBox(0.3, 0.5, 0.1, 0.05, text="MH12AB1234"), PlateBox(0.7, 0.5, 0.1, 0.05)),
        )
    ]
    assert compute_stats(records).with_text == 1


def test_split_breakdown_matches_the_assignment():
    records = [
        ImageRecord(f"clip{g}_frame_{f}", 100, 100, boxes=(PlateBox(0.5, 0.5, 0.1, 0.05),))
        for g in range(12)
        for f in range(4)
    ]
    assignment = split_records(records)
    stats = compute_stats(records, assignment)
    assert sum(stats.by_split.values()) == len(records)
    assert sum(stats.plates_by_split.values()) == stats.plates
    assert set(stats.by_split) == {s.value for s in Split}


def test_source_breakdown_tracks_licence_provenance():
    # Sources carry different licences; this is what lets us drop the
    # non-redistributable ones before publishing a derived dataset.
    records = [
        ImageRecord("a", 100, 100, source="uc3m-lp"),
        ImageRecord("b", 100, 100, source="uc3m-lp"),
        ImageRecord("c", 100, 100, source="roboflow-us-eu"),
    ]
    assert compute_stats(records).by_source == {"uc3m-lp": 2, "roboflow-us-eu": 1}


class TestExitCriteria:
    # The default asks for IN + EU, not IN + DE: the openly-licensed European
    # dataset is pan-European, and requiring a DE tag would fail on data that
    # trains the detector perfectly well. German plates are Phase 5's grammar.
    def _stats(self, n_plates, regions=(Region.INDIA, Region.EUROPE)):
        records = []
        for i in range(n_plates):
            region = regions[i % len(regions)]
            records.append(
                ImageRecord(
                    f"img{i}", 1000, 1000, boxes=(PlateBox(0.5, 0.5, 0.1, 0.05, region=region),)
                )
            )
        return compute_stats(records)

    def test_passes_when_all_criteria_met(self):
        assert check_exit_criteria(self._stats(1000)) == []

    def test_reports_insufficient_plates(self):
        failures = check_exit_criteria(self._stats(10))
        assert any("need >= 1000" in f for f in failures)

    def test_reports_a_missing_region(self):
        failures = check_exit_criteria(self._stats(1000, regions=(Region.INDIA,)))
        assert any("region EU" in f for f in failures)

    def test_required_regions_are_configurable(self):
        # Phase 5 work, or a later German capture, can demand a DE tag.
        failures = check_exit_criteria(self._stats(1000), required_regions=(Region.GERMANY,))
        assert any("region DE" in f for f in failures)

    def test_reports_every_failure_not_just_the_first(self):
        # The notebook should print the whole picture rather than stopping at
        # the first shortfall.
        failures = check_exit_criteria(self._stats(10, regions=(Region.INDIA,)))
        assert len(failures) >= 2

    def test_threshold_is_configurable(self):
        assert check_exit_criteria(self._stats(10), min_plates=5) == []
